# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:12:30 2023

@author: tsdan
"""

import os
import shutil
import pathlib
import random
import sys
from time import sleep
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2
from skimage import data
from skimage.color import rgb2hsv
import pandas as pd
os.chdir('path_working_directory')
img_size = 512
img = Image.open('image_à_filtrer')
img_resized = img.resize((img_size,img_size))
img_array = np.array(img_resized)

def HSV(image):
    imagecolor = np.asarray(image)
    imagehsv = rgb2hsv(imagecolor)
    H = imagehsv[:,:,0]*255
    S = imagehsv[:,:,1]*255
    V = imagehsv[:,:,2]*255
    return([H,S,V])

def green_index_pixel(image_array,image,i,j):
    red = image_array[0]
    green = image_array[1]
    blue = image_array[2]
    r = red/(red+green+blue)
    g = green/(red+green+blue)
    b = blue/(red+green+blue)
    ExG = 2*g - r -b
    M_ExG = (-0.884 * r) + (1.262 * g) + (0.311 * b)
    ExR = (1.4 * r) - g
    CIVE = (0.441 * r) - (0.811 + g) + (0.385 * b) + 18.787
    VEG = g/((r**0.667)*(b**0.333))
    HSV_parameters = HSV(image)
    if (HSV_parameters[0][i,j] < 50) or (HSV_parameters[0][i,j]>150) or (HSV_parameters[2][i,j] > 49):
        HSVDT = 0
    else:
        HSVDT = 1
    return([ExG,M_ExG, ExR, CIVE, VEG, HSVDT])


def find_max_index(image_array,image):
    max_ExG = 0
    max_M_ExG = 0
    max_ExR = 0
    max_CIVE = 0
    max_VEG = 0
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            pixel = image_array[i,j]
            liste_index = green_index_pixel(pixel,image,i,j)
            if liste_index[0] > max_ExG:
                max_ExG = liste_index[0]
            if liste_index[1] > max_M_ExG:
                max_M_ExG = liste_index[1]
            if liste_index[2] > max_ExR:
                max_ExR = liste_index[2]
            if liste_index[3] > max_CIVE:
                max_CIVE = liste_index[3]
            if liste_index[4] > max_VEG:
                max_VEG = liste_index[4]
    return([max_ExG, max_M_ExG, max_ExR, max_CIVE, max_VEG])
        

def RGB(pixel):
    red = int(pixel[0])
    green = int(pixel[1])
    blue = int(pixel[2])
    all_rgb = int(red+green+blue)
    if all_rgb == 0:
        (r,g,b) = (0,0,0)
    else:
        r = red/all_rgb
        g = green/all_rgb
        b = blue/all_rgb
    return([r,g,b])

def calcul_ExG(liste_rgb):
    r = liste_rgb[0]
    g = liste_rgb[1]
    b = liste_rgb[2]
    valeurExG = (2*g - r - b)
    return(valeurExG)

def calculMExG(liste_rgb):
    r = liste_rgb[0]
    g = liste_rgb[1]
    b = liste_rgb[2]
    valeurMExG = (-0.884*r) + (1.262*g) - (0.311*b)
    return(valeurMExG)



def ExG(array):
    ExG_array = np.zeros(shape = (array.shape[0],array.shape[1]))
    binary_ExG = np.zeros(shape = (array.shape[0],array.shape[1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ExG_array[i,j] = calcul_ExG(RGB(array[i,j]))
    max_ExG = np.max(ExG_array)
    for i in range(ExG_array.shape[0]):
        for j in range(ExG_array.shape[1]):
            if ExG_array[i,j] >= max_ExG/10:
                binary_ExG[i,j] = 1
    return(binary_ExG)

def M_ExG(array):
    MExG_array = np.zeros(shape = (array.shape[0],array.shape[1]))
    binary_MExG = np.zeros(shape = (array.shape[0],array.shape[1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            MExG_array[i,j] = calculMExG(RGB(array[i,j]))
    max_MExG = np.max(MExG_array)
    for i in range(MExG_array.shape[0]):
        for j in range(MExG_array.shape[1]):
            if MExG_array[i,j] >= max_MExG/10:
                binary_MExG[i,j] = 1
    return(binary_MExG)


def calcul_ExR(liste_rgb):
    r = liste_rgb[0]
    g = liste_rgb[1]
    valeurExR = 1.4*r - g
    return(valeurExR)

def ExR(array):
    ExR_array = np.zeros(shape = (array.shape[0],array.shape[1]))
    binary_ExR = np.zeros(shape = (array.shape[0],array.shape[1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ExR_array[i,j] = calcul_ExR(RGB(array[i,j]))
    max_ExR = np.max(ExR_array)
    for i in range(ExR_array.shape[0]):
        for j in range(ExR_array.shape[1]):
            if ExR_array[i,j] >= max_ExR/10:
                binary_ExR[i,j] = 1
    binary_ExR = np.ones(shape = (array.shape[0],array.shape[1])) - binary_ExR
    return(binary_ExR)



def calcul_VEG(liste_rgb):
    r = liste_rgb[0]
    g = liste_rgb[1]
    b = liste_rgb[2]
    if (r**0.667)*(b**0.333) ==0:
        valeurVEG = g
    else:
        valeurVEG = g/((r**0.667)*(b**0.333))
    return(valeurVEG)

def VEG(array):
    VEG_array = np.zeros(shape = (array.shape[0],array.shape[1]))
    binary_VEG = np.zeros(shape = (array.shape[0],array.shape[1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            VEG_array[i,j] = calcul_VEG(RGB(array[i,j]))
    max_VEG = np.max(VEG_array)
    for i in range(VEG_array.shape[0]):
        for j in range(VEG_array.shape[1]):
            if VEG_array[i,j] >= max_VEG/10:
                binary_VEG[i,j] = 1
    return(binary_VEG)



def calcul_CIVE(liste_rgb):
    r = liste_rgb[0]
    g = liste_rgb[1]
    b = liste_rgb[2]
    valeurCIVE = (0.441 * r) - (0.811 + g) + (0.385 * b) + 18.787
    return(valeurCIVE)

def normalize(value,mean,std):
    norm_value = (value-mean)/std
    return(norm_value)

def CIVE(array):
    CIVE_array = np.zeros(shape = (array.shape[0],array.shape[1]))
    binary_CIVE = np.zeros(shape = (array.shape[0],array.shape[1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            CIVE_array[i,j] = calcul_CIVE(RGB(array[i,j]))
    mean_CIVE = np.mean(CIVE_array)
    std_CIVE = np.std(CIVE_array)
    for i in range(CIVE_array.shape[0]):
        for j in range(CIVE_array.shape[1]):
            CIVE_array[i,j] = normalize(CIVE_array[i,j], mean_CIVE, std_CIVE)
            if CIVE_array[i,j] < 0:
                binary_CIVE[i,j] = 1
    return(binary_CIVE)


def calcul_HSV(image):
    imagecolor = np.asarray(image)
    imagehsv = rgb2hsv(imagecolor)
    H = imagehsv[:,:,0]*255
    S = imagehsv[:,:,1]*255
    V = imagehsv[:,:,2]*255
    return([H,S,V])

def HSVDT(image,array):
    binary_HSV = np.zeros(shape = (array.shape[0],array.shape[1]))
    HSV_parameters = calcul_HSV(image)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if (HSV_parameters[0][i,j] < 50) or (HSV_parameters[0][i,j]>150):
                binary_HSV[i,j] = 0
            else:
                binary_HSV[i,j] = 1
    return(binary_HSV)

def metaindex(array,image):
    ExG_ = ExG(array)
    M_ExG_ = M_ExG(array)
    ExR_ = ExR(array)
    VEG_ = VEG(array)
    CIVE_ = CIVE(array)
    HSVDT_ = HSVDT(image,array)
    MI = ExG_ + M_ExG_ + ExR_ + VEG_ + CIVE_ + HSVDT_
    binary_MI = np.zeros(shape = (MI.shape[0],MI.shape[1]))
    for i in range(binary_MI.shape[0]):
        for j in range(binary_MI.shape[1]):
            if MI[i,j] > 3:
                binary_MI[i,j] = 1
    array[:,:,0] = array[:,:,0]*binary_MI
    array[:,:,1] = array[:,:,1]*binary_MI
    array[:,:,2] = array[:,:,2]*binary_MI
    return(array)


filter_array = metaindex(img_array,img_resized)
filter_image = Image.fromarray(filter_array)
plt.imshow(filter_image)

      
    

              

            
    



    
    



      
    

              

            
    



    
    

    
    
    
    
    
        
