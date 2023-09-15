import torch
import albumentations as A
import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 512  # 3000 originally
IMAGE_WIDTH = 512  # 4000 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = '/content/drive/MyDrive/Colab_Notebooks/STAGE/COCO_dataset/data/train_images/'
TRAIN_MASK_DIR = '/content/drive/MyDrive/Colab_Notebooks/STAGE/COCO_dataset/data/train_masks/'
VAL_IMG_DIR = '/content/drive/MyDrive/Colab_Notebooks/STAGE/COCO_dataset/data/val_images/'
VAL_MASK_DIR = '/content/drive/MyDrive/Colab_Notebooks/STAGE/COCO_dataset/data/val_masks/'


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=180, p=1.0),
            A.ColorJitter(brightness=(0.90,1.1),contrast=(1),saturation=(0.95,1.05),hue=(-0.01,0.01)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    dice_list =[]
    IOU_list = []
    epochs = []
    df = pd.DataFrame({"dice_score" : dice_list,
                      "IOU_score" : IOU_list,
                      "epochs" : epochs})
                      
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        df = pd.read_csv('saved_images/dataframe.csv', sep=';')


    scaler = torch.cuda.amp.GradScaler()
    check_accuracy(val_loader, model, device=DEVICE)
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)
        
        # print some examples to a folder
        dice_mean = []
        IOU_mean = []
        save_predictions_as_imgs(
            val_loader, model, dice_mean, IOU_mean, folder="saved_images/", device=DEVICE
        )
        
        dice = np.mean(dice_mean)
        IOU = np.mean(IOU_mean)
        new_data = {'dice_score': dice, 'IOU_score': IOU, 'epochs': len(df)}
        df.loc[len(df)] = new_data
        df.to_csv('saved_images/dataframe.csv', sep=';', index=False, encoding='utf-8')
    res = df.plot(x="epochs",y="dice_score").get_figure()
    res.savefig('saved_images/graph.jpg')
    res2 = df.plot(x="epochs",y="IOU_score").get_figure()
    res2.savefig('saved_images/graph2.jpg')


if __name__ == "__main__":
    main()