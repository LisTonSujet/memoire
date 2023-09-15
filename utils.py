# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:23:50 2023

@author: tsdan
"""

import torch
import torchvision
from dataset import FevDataset
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = FevDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = FevDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")

def save_predictions_as_imgs(
    loader, model, dice_mean, IOU_mean, folder="saved_images/", device="cuda"
):
    path = '/content/drive/MyDrive/Colab_Notebooks/STAGE/COCO_dataset/'
    model.eval()
    
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        pred = Image.open(os.path.join(path,f"saved_images/pred_{idx}.png")).convert('L')
        source = Image.open(os.path.join(path,f"saved_images/{idx}.png")).convert('L')
        green = np.array(source)
        red = np.array(pred)
        blue = np.zeros((green.shape[0],green.shape[1]), np.uint8)
        blank_image = np.zeros((green.shape[0],green.shape[1],3), np.uint8)
        blank_image = (np.dstack((red,green,blue)) * 255.999) .astype(np.uint8)
        TN = (blank_image == [0, 0, 0]).all(axis=2).sum()
        FN = (blank_image == [0, 255, 0]).all(axis=2).sum()
        TP = (blank_image == [255, 255, 0]).all(axis=2).sum()
        FP = (blank_image == [255, 0, 0]).all(axis=2).sum()
        IOU_score = TP/(TP+FP+FN)
        dice_score = 2*TP/(2*TP + FP + FN)
        image = Image.fromarray(blank_image)
        image.save(f"saved_images/compare_{idx}.jpg","JPEG")
        dice_mean.append(dice_score)
        IOU_mean.append(IOU_score)
        