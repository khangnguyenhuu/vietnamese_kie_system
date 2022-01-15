import io
import cv2
import numpy as np
from PIL import Image

from app.src.pannet_predict import PanNet
from app.src.vietocr_predict import VietOCR
from app.src.sdmgr_predict import SDMGR

TextDet = PanNet(device="cuda:0")
TextRecog = VietOCR()
KIE = SDMGR(device="cuda:0")

image = cv2.imread("./Z71pofS2IRrQD-R9W7Unqj2wZ-ikE9Ovmh45bZKe1qY-2.png")

bboxes_2_points, bboxes_4_points = TextDet.detect(image)
print("2 points", bboxes_2_points)
print("4 points", bboxes_4_points)
box_feed_to_kie = []
for i, box in enumerate(bboxes_2_points):
    line = image[box[1]:box[3],box[0]:box[2]]
    line = Image.fromarray(line)
    text = TextRecog.recognize(line)
    x1, y1, x2, y2, x3, y3, x4, y4 = bboxes_4_points[i]
    box_feed_to_kie.append([x1, y1, x2, y2, x3, y3, x4, y4, text])
kie_result = KIE.extract_infor(image, box_feed_to_kie)
print(kie_result)