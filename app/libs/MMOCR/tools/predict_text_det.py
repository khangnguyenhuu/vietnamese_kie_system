from mmocr.utils.ocr import MMOCR
import time
import glob
import cv2
import PIL
import numpy as np
# Load models into memory
ocr = MMOCR(det='PANet_CTW', recog=None)
# process bbox

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

# Inference
count_time_1= time.time()
for img_path in glob.glob("/content/MMOCR-copy/data/imgs/test/*"):

    img = cv2.imread(img_path)
    list_box_results = ocr.readtext(img, output='hello.jpg', export='./')
    list_box_results = convert_xyminmax(list_box_results)
    print(list_box_results)
print("Time processing: ",time.time() - count_time_1)
