import cv2
import numpy as np
from PIL import Image

from src.pannet.pannet_predict import PanNet
from src.vietocr.vietocr_predict import VietOCR
from utils import get_config, loadImage

if __name__ == '__main__':
    img_path = "demo.png"
    img = cv2.imread(img_path)
    box_2_points, box_4_points = PanNet.detect(img)
    contents=[]
    for box in box_2_points:
        line = img[box[1]:box[3],box[0]:box[2]]
        line = Image.fromarray(line)
        text = TextRecog.recognize(line)
        contents.append(text)
        print("boxs:",box)
        print("contents:",text)
