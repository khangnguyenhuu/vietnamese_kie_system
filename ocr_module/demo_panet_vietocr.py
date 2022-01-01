from libs.MMOCR.mmocr.utils.ocr import MMOCR
import cv2
import numpy as np
from PIL import Image
from libs.vietocr.vietocr.tool.predictor import Predictor
from libs.vietocr.vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


from libs.MMOCR.mmocr.apis import init_detector
from libs.MMOCR.mmocr.utils.model import revert_sync_batchnorm
from utils import get_config, loadImage, tlwh_2_maxmin

# load model
# Panet's model 
ocr=MMOCR()
device='cuda:0'
det_config = "/content/MMOCR-copy/configs/textdet/panet/panet_r18_fpem_ffm_600e_ctw1500.py"
det_ckpt = "/content/MMOCR-copy/panet/epoch_10.pth"
detect_model = init_detector(det_config, det_ckpt, device=device)
detect_model = revert_sync_batchnorm(detect_model)


# VietOCR's model
config_reg = Cfg.load_config_from_file("config.yml")
reg_model = Predictor(config_reg)

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

def predict_detection(img,ocr,detect_model):
    list_box_results = ocr.readtext(img,detect_model=detect_model)
    list_box_results = convert_xyminmax(list_box_results)
    return list_box_results

if __name__ == '__main__':
    img_path = "demo.png"
    img = cv2.imread(img_path)
    boxs = predict_detection(img,ocr,detect_model)
    contents=[]
    for box in boxs:
        line = img[box[1]:box[3],box[0]:box[2]]
        line = Image.fromarray(line)
        text = reg_model.predict(line)
        contents.append(text)
        print("boxs:",box)
        print("contents:",text)
