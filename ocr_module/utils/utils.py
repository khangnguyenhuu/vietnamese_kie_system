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

def tlwh_2_maxmin(bboxes):
    new_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        new_bboxes.append([xmin, ymin, xmax, ymax])
    new_bboxes = np.array(new_bboxes)
    return new_bboxes


def json2txt (json_path, txt_folder):
    '''
    args: json_path: path to json output file
          txt_folder: path to txt folder when convert done
    return:
          convert json output file to txt file to evaluate mAP
    '''
    with open(json_path, "r") as fp:
        if os.path.exists(txt_folder) == False:
            os.mkdir(txt_folder)
        data = json.load(fp)
        for image in data:
            txt_file = open (os.path.join(txt_folder, image['image_name']) + ".txt", "a+")
            for result in image['result']:
                txt_file.write("{} {} {} {} {} {} {}".format(result['text'], "1", result['x_min'], result['y_min'], \
                                                                               result['x_max'], result['y_max'], "\n"))

