import io
import cv2
import numpy as np
import time
from PIL import Image

from fastapi import APIRouter, Form, File, UploadFile
from ..src.pannet_predict import PanNet
from ..src.vietocr_predict import VietOCR
from ..src.sdmgr_predict import SDMGR
from ..utils.logger import logger

try:
  logger.info(f'[LOADING] Text detection model')
  start = time.time()
  TextDet = PanNet(device="cuda:0")
  end = time.time()
  logger.info(f'[LOADING SUCESS] Text detection model')
  logger.info('[PROCESS TIME] {}'.format(end-start))
except:
  logger.error(f'[ERROR LOADING] Text detection model')

#try:
logger.info(f'[LOADING] Text recognition model')
start = time.time()
TextRecog = VietOCR()
end = time.time()
logger.info(f'[LOADING SUCESS] Text recognition model')
logger.info('[PROCESS TIME] {}'.format(end-start))
#except:
 # logger.error(f'[ERROR LOADING] Text recognition model')

try:
  logger.info(f'[LOADING] Key information extraction model')
  start = time.time()
  KIE = SDMGR(device="cuda:0")
  end = time.time()
  logger.info(f'[LOADING SUCESS] Key information extraction model')
  logger.info('[PROCESS TIME] {}'.format(end-start))
except:
  logger.error(f'[ERROR LOADING] Key information extraction model')

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
    logger.info(f'[START PROCESSING] Detect text')
    start = time.time()
    bboxes_2_points, bboxes_4_points = TextDet.detect(image)
    end = time. time()
    logger.info(f'[END PROCESSING] Detect text')
    logger.info('[TEXT DETECTION PROCESS TIME] {}'.format(end-start))
    
    box_feed_to_kie = []
    logger.info(f'[START PROCESSING] Recognize text')
    start = time.time()
    for i, box in enumerate(bboxes_2_points):
        line = image[box[1]:box[3],box[0]:box[2]]
        line = Image.fromarray(line)
        text = TextRecog.recognize(line)
        x1, y1, x2, y2, x3, y3, x4, y4 = bboxes_4_points[i]
        box_feed_to_kie.append([x1, y1, x2, y2, x3, y3, x4, y4, text])
    end = time.time()
    logger.info(f'[END PROCESSING] Recognize text')
    logger.info('[TEXT RECOGNITION PROCESS TIME] {}'.format(end-start))
    
    logger.info(f'[START PROCESSING] Extract information from document')
    start = time.time()
    kie_result = KIE.extract_infor(image, box_feed_to_kie)
    end = time.time()
    logger.info(f'[END PROCESSING] Extract information from document')
    logger.info('[KIE TIME PROCESS TIME] {}'.format(end-start))
    
    return {
        'status': '200',
        'message': 'OK',
        'data': {
            'bboxes': kie_result
        }
    }
