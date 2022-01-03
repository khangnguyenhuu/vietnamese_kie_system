'''
author: khangnh
created: 21/07/2021
last modified: 18/06/2021
'''
import os
import cv2
import json
import numpy as np 

def loadImage(img_file):
    img = cv2.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def convert_xyminmax(list_box):
    new_list = []
    for box in list_box[0]["boundary_result"]:
        box = box[:-1]
        xmin = int(min(box[0::2])-5)
        xmax = int(max(box[0::2])+5)
        ymin = int(min(box[1::2])-5)
        ymax = int(max(box[1::2])+5)
        new_list.append([xmin, ymin, xmax, ymax])
    
    return new_list