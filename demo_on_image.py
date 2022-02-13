import io
import cv2
import os
import numpy as np
from PIL import Image

from app.src.pannet_predict import PanNet
from app.src.vietocr_predict import VietOCR
from app.src.sdmgr_predict import SDMGR
import time

TextDet = PanNet(device="cuda:0")
TextRecog = VietOCR()
KIE = SDMGR(device="cuda:0")
src = "./test/images"
fps = 0
count = 0
for i in os.listdir(src):
  path = os.path.join(src, i)
  name = i.replace(".png", "")
  with open("./output/" + name +".txt", "a+", encoding="utf-8") as fp1:
    image = cv2.imread(path)
    start = time.time()
    count += 1
    bboxes_2_points, bboxes_4_points = TextDet.detect(image)
    box_feed_to_kie = []
    for i, box in enumerate(bboxes_2_points):
        try:
          line = image[box[1]:box[3],box[0]:box[2]]
          line = Image.fromarray(line)
          text = TextRecog.recognize(line)
          x1, y1, x2, y2, x3, y3, x4, y4 = bboxes_4_points[i]
          box_feed_to_kie.append([x1, y1, x2, y2, x3, y3, x4, y4, text])
        except:
          pass
    kie_result = KIE.extract_infor(image, box_feed_to_kie)
    end = time.time()
    process_time = end - start
    fps+=process_time
    for line in kie_result:
        print(line[0])
        print(line[1][-1])
        save =  line[1][-1] + ", " + line[0] + "\n"
        fp1.write(save)

print("AVG time process: ", fps/count)
