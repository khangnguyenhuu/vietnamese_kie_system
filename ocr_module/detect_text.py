import os 
import time

import cv2
import torch
import argparse
import numpy as np
import json

from utils import get_config, loadImage, tlwh_2_maxmin

# from libs.CRAFT.craft import CRAFT
os.environ['CUDA_VISIBLE_DEVICES']='4'

# setup gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextDetection():
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.output_dict = []
        print(self.cfg.PIPELINE.TEXT_DETECTION)
        if self.cfg.PIPELINE.TEXT_DETECTION.CRAFT:
            from src.CRAFT.craft_predict import MyCRAFT
            self.model = MyCRAFT(self.cfg.CRAFT)
        if self.cfg.PIPELINE.TEXT_DETECTION.MMOCR:
            from src.MMOCR.mmocr_predict import MyMMOcr
            self.model = MyMMOcr(self.cfg.MMOCR)

    def detection(self, image):
        '''
        This function will run detection method on single image
        Args: 
            image (array): image for predict
        Return:
            bboxes (array): list of array
                [[x1, y1, x2, y2, x3, y3, x4, y4], ...]
        '''
        # if self.cfg.PIPELINE.TEXT_DETECTION.CRAFT:
        bboxes = self.model.detect(image)
        return bboxes

    def run_detection_on_folder(self, folder_path):
        '''
        This function will run detection on folder (list of images)
        and auto save at detection in path args.detection_output_path
        '''
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            bboxes = self.detection(image=image)
            print ('Detect on image: {}'.format(image_name))
            self.write_detection_result(image_name, bboxes)

    def write_detection_result(self, image_name, bboxes):
        path = os.path.join(self.args.detection_output_path, image_name.split('.')[0] + '.txt')
        f = open(path, 'w+')
        for bbox in bboxes:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1], bbox[2][0], bbox[2][1], bbox[3][0], bbox[3][1]
            f.write('{} {} {} {} {} {} {} {}{}'.format(str(x1), str(y1), str(x2),
                                                        str(y2), str(x3), str(y3),
                                                        str(x4), str(y4), '\n'))
        print ('File save at: ', path)
        f.close()

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_on_folder', type=bool, default=False,
                        help='Wheter or not run on folder, if false, it will auto run on image and image_path will require')
    parser.add_argument("-i", "--image_path", type=str,
                        help='Path to image')
    parser.add_argument("--folder_path", type=str, default='./data',
                        help='Path to folder')
    parser.add_argument("--run_detection", type=bool, default=True)
    parser.add_argument("--detection_output_path", type=str, default='./output_detection',
                        help='Path to detection output, detection output will save as txt file in this folder')
    parser.add_argument("--detection_weight", type=str, default='./models/craft_model.pth',
                        help='Path to detection weight')
    parser.add_argument("--config_method", type=str, default='./configs/craft.yaml',
                        help='Path to detection config method')
    parser.add_argument("--config_pipeline", type=str, default='./configs/pipeline.yaml',
                        help='Path to pipeline config method')
    return parser.parse_args()
if __name__=='__main__':

    args = parse_args()
    # setup config
    cfg = get_config()
    cfg.merge_from_file(args.config_pipeline)
    cfg.merge_from_file(args.config_method)

    if (os.path.exists(args.detection_output_path) == False):
        os.mkdir(args.detection_output_path)

    text_detection = TextDetection(cfg, args)

    if (args.run_on_folder == False):
        if (args.image_path == None):
            print ("Can't find path to image, please use --image_path arguments")
            exit()
        else:
            image = cv2.imread(args.image_path)
            text_detection.detection(image)
    else:
        text_detection.run_detection_on_folder(args.folder_path)
