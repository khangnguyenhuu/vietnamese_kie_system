import io
import cv2
import numpy as np
from PIL import Image

from fastapi import APIRouter, Form, File, UploadFile
from ..src.pannet_predict import PanNet
from ..src.vietocr_predict import VietOCR
from ..src.sdmgr_predict import SDMGR

TextDet = PanNet(device="cuda:0")
TextRecog = VietOCR()
KIE = SDMGR(device="cuda:0")

router = APIRouter()

@router.post('/kie')
def detect(file: UploadFile = File(...)):
    # Read image from bytes
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert('RGB')
    except:
        return {
            'code': '1001',
            'status': 'Error while parsing image to RGB image'
        }
    # Convert image to BGR
    image = np.asarray(image)[:, :, ::-1]

    bboxes_2_points, bboxes_4_points = TextDet.detect(image)
    box_feed_to_kie = []
    for i, box in enumerate(bboxes_2_points):
        line = image[box[1]:box[3],box[0]:box[2]]
        line = Image.fromarray(line)
        text = TextRecog.recognize(line)
        x1, y1, x2, y2, x3, y3, x4, y4 = bboxes_4_points[:]
        box_feed_to_kie.append([x1, y1, x2, y2, x3, y3, x4, y4, text])
    kie_result = KIE.extract_infor(image, bboxes_text)
    return {
        'code': '1000',
        'data': {
            'bboxes': kie_result
        }
    }