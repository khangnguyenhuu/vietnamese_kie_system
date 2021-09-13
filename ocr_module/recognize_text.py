import os 
import time

import cv2
import torch
import argparse
import numpy as np
import json

from utils import get_config, loadImage, tlwh_2_maxmin

# from libs.vietocr.tool.config import Cfg

#from src.MORAN.moran_predict import myMoran
#from src.vietocr.vietocr_predict import myVietOCR

os.environ['CUDA_VISIBLE_DEVICES']='4'
# setup gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'

class TextRecognition:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.output_dict = []
        # text recognition model
        if (self.cfg.PIPELINE.TEXT_RECOGNITION.MORAN==True):
            from src.MORAN.moran_predict import myMoran
            self.model = myMoran(self.cfg.MORAN, self.args.recognition_weight)
        if (self.cfg.PIPELINE.TEXT_RECOGNITION.VIETOCR==True):
            from src.vietocr.vietocr_predict import myVietOCR
            from libs.vietocr.tool.config import Cfg
            config = Cfg.load_config_from_file(args.config_method)
            self.model = myVietOCR(config)
        if self.cfg.PIPELINE.TEXT_RECOGNITION.DEEPTEXT==True:
            from src.deeptext.deeptext_predict import myDeepText
            self.model = myDeepText(self.cfg.DEEPTEXT, self.args.recognition_weight)
    

    def recognition(self, image=None):
        '''
        This function will run recognition with single image
        Args: 
            image (np.array)
        Return:
            text (str): Text in this image (auto upper)
        '''
        text = self.model.recognize(image)
        text.upper()
        return text

    def recognition_on_region_of_image(self, image, x_min, y_min, x_max, y_max):
        '''
        This function will run recognition in a region (can load value from 
        detection result)
        Args:
            image (np.array)
            x_min
            y_min
            x_max
            y_max
        Return: Text of this region
        '''
        x_min, y_min, x_max, y_max = float(x_min), float(y_min), float(x_max), float(y_max)
        region = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        cv2.imwrite('./region.jpg', region)
        text = self.recognition(image=region)
        return text

    def run_recognition_on_folder(self, folder_path):
        '''
        This function will run recognition in folder have 2 modes:
        1. Run with detection result (need folder output_detection as folder_path input)
        2. Run with normal folder (image as input)
        Args: 
            Folder_path (str): Path to folder
        Return:
            With mode 1: result save at full_pipeline_output and with json format
            With mode 2: will print the name of image and text of this image
        '''
        if (self.args.run_recognition_with_detection_result == True):
            print("Beacause config run on detection result is True so the output json file will save at: ", self.args.full_pipeline_output_path)
            if (os.path.exists(args.recognition_output_path) == False):
                os.mkdir(args.recognition_output_path)  
            for path in os.listdir(args.detection_output_path):
                image_name = path.split('.')[0]
                print ('image name: ', image_name)
                path = os.path.join(args.detection_output_path, path)
                f = open(path, 'r')
                bboxes = f.read()
                bboxes = [row.split(' ') for row in bboxes.split('\n')][:-2]
                f.close()

                image_path = os.path.join(folder_path, image_name)
                if (os.path.exists(image_path+'.jpg')):
                    image = cv2.imread(image_path+'.jpg')
                else:
                    image = cv2.imread(image_path+'.png')
                # record list to save result
                record = {}
                record['image_name'] = image_name
                self.objs = []
                for bbox in bboxes:
                    text = self.recognition_on_region_of_image(image=image,
                                                                x_min=bbox[0], y_min=bbox[1],
                                                                x_max=bbox[4], y_max=bbox[5])
                    print ('x_min: {}, y_min: {}, x_max: {}, y_max: {}, text: {}'.format(bbox[0], bbox[1], bbox[4], bbox[5], text))
                    self.append_objs(bbox[0], bbox[1], bbox[4], bbox[5], text)
                record['result'] = self.objs
                self.output_dict.append(record)
        else:
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = cv2.imread(image_path)
                text = self.recognition(image=image)
                print('image name: , text: '.format(image_name, text))
        self.write_file_result()

    def append_objs(self, x_min, y_min, x_max, y_max, text):
        obj = {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'text': text
        }
        self.objs.append(obj)
        
    def write_file_result(self):
        with open(os.path.join(args.full_pipeline_output_path, 'output.json'), 'w') as f:
            json.dump(self.output_dict, f)

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_on_folder', type=bool, default=False,
                        help='Wheter or not run on folder, if False auto run on single image and need args image_path')
    parser.add_argument("-i", "--image_path", type=str,
                        help='Path to image')
    parser.add_argument("--folder_path", type=str, default='./data',
                        help='Path to folder will run predict, it not require if using detection result as input')
    parser.add_argument("--run_recognition", type=bool, default=True)
    parser.add_argument("--recognition_output_path", type=str, default='./output_recognition',
                        help='Path to save recognition result, require if run without detection result as input')
    parser.add_argument("--recognition_weight", type=str, default='./models/moranv2_model.pth',
                        help='path to recognition method weight')
    parser.add_argument("--run_recognition_with_detection_result", type=bool, default=False,
                        help='Wheter or not use detection result as input for recognition step (detetion result will store at txt file)')
    parser.add_argument("--detection_output_path", type=str, default='./output_detection', 
                        help='path to detection output, require if run_recognition_with_detection_result True, it will use detection result as input')
    parser.add_argument("--full_pipeline_output_path", type=str, default='./output_full_pipeline',
                        help='If args run_recognition_with_detection_result is True, output will save as json file in this folder')
    parser.add_argument("--config_method", type=str, default='./configs/MORANv2.yaml',
                        help='Path to recognition config method')
    parser.add_argument("--config_pipeline", type=str, default='./configs/pipeline.yaml',
                        help='Path to full pipeline config method')

    return parser.parse_args()

if __name__=='__main__':

    args = parse_args()
    print(args)
    # setup config
    cfg = get_config()
    cfg.merge_from_file(args.config_pipeline)
    cfg.merge_from_file(args.config_method)

    if (os.path.exists(args.full_pipeline_output_path) == False):
        os.mkdir(args.full_pipeline_output_path)
    if (os.path.exists(args.recognition_output_path) == False):
        os.mkdir(args.recognition_output_path)

    text_recognition = TextRecognition(cfg, args)

    if (args.run_on_folder == False):
        if (args.image_path == None):
            print ("Can't find path to image, please use --image_path arguments")
            exit()
        else:
            image = cv2.imread(args.image_path)
            text_recognition.recognition(image)
    else: 
        text_recognition.run_recognition_on_folder(args.folder_path)
