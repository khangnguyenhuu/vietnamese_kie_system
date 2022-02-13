import io
import cv2
import os
import numpy as np
from PIL import Image

from app.src.pannet_predict import PanNet
from app.src.vietocr_predict import VietOCR
from app.src.sdmgr_predict import SDMGR
import time

#TextDet = PanNet(device="cuda:0")
#TextRecog = VietOCR()
KIE = SDMGR(device="cuda:0")
src = "./test/images"
src_box = "./test/boxes_trans"
fps = 0
count = 0
for i in os.listdir(src):
  path = os.path.join(src, i)
  name = i.replace(".png", "")
  with open("./output/" + name +".txt", "a+", encoding="utf-8") as fp1:
    with open(os.path.join(src_box, name + ".tsv"), "r+", encoding = "utf-8") as fp2:
      image = cv2.imread(path)
      line = fp2.readlines()
      start = time.time()
      count += 1
      box_feed_to_kie = []
      for i, box in enumerate(line):
        box = box.split(",")
        text = box[-2]
        x1, y1, x2, y2, x3, y3, x4, y4 = box[1:-2]
        box_feed_to_kie.append([int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4), text])
        print([int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4), text])

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
