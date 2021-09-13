'''
author: khangnh
last modified: 18/06/2021
note: this file is no longer update
'''
import os 
import time

import cv2
import torch
import argparse
import numpy as np
import json

from utils import get_config, loadImage, tlwh_2_maxmin

from libs.CRAFT.craft import CRAFT

from src.CRAFT.craft_predict import craft_text_detect, load_model_craft
from src.MORAN.moran_predict import moran_text_recog, load_model_moran
# setup gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageRecognition():
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.output_dict = []
        # text detection model 
        if (self.cfg.PIPELINE.TEXT_DETECTION.CRAFT == True):
            self.CRAFT_CONFIG = cfg.CRAFT
            self.NET_CRAFT = CRAFT()
            print('[LOADING] TEXT DETECTION MODEL')
            self.CRAFT_MODEL = load_model_craft(self.CRAFT_CONFIG, self.NET_CRAFT, self.args.detection_weight)
            print ('[LOADING SUCESS] TEXT DETECTION MODEL')
        # text recognition model
        if (self.cfg.PIPELINE.TEXT_RECOGNITION.MORAN== True):
            self.MORAN_CONFIG = cfg.MORAN
            print('[LOADING] TEXT RECOGNITION MODEL')
            self.MORAN_TRANSFORMER, self.MORAN_CONVERTER, self.MORAN = load_model_moran(cfg=self.MORAN_CONFIG, 
                                                                                        model_weight_path=self.args.recognition_weight)
            print ('[LOADING SUCESS] TEXT RECOGNITION MODEL')

    def detection(self, image, image_name=None):
        '''
        This function will run detection method on single image
        Args: 
            image (array): image for predict
            image_name (str): name of image run detection
        Return:
            bboxes (array): list of array
                [[x1, y1, x2, y2, x3, y3, x4, y4], ...]
        '''
        if (self.cfg.PIPELINE.TEXT_DETECTION.CRAFT == True):
            bboxes, polys, score = craft_text_detect(image, self.CRAFT_CONFIG, self.CRAFT_MODEL)
        if (self.args.save_visualize == True):
            self.visualize(image, bboxes_detection=bboxes, image_name=image_name)
            return bboxes

    
    def run_detection_on_folder(self, folder_path):
        '''
        This function will run detection on folder (list of images)
        and auto save at detection in path args.detection_output_path
        '''
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            bboxes = self.detection(image=image, image_name=image_name)
            print ('Detect on image: {}'.format(image_name))
            self.write_detection_result(image_name, bboxes)

    def write_detection_result(self, image_name, bboxes):
        '''
        utils for save detection result
        '''
        path = os.path.join(self.args.detection_output_path, image_name.split('.')[0] + '.txt')
        f = open(path, 'w+')
        for bbox in bboxes:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1], bbox[2][0], bbox[2][1], bbox[3][0], bbox[3][1]
            f.write('{} {} {} {} {} {} {} {}{}'.format(str(x1), str(y1), str(x2),
                                                        str(y2), str(x3), str(y3),
                                                        str(x4), str(y4), '\n'))
        print ('File save at: ', path)
        f.close()

    def recognition(self, image):
        '''
        This function will run recognition with single image
        Args: 
            image (np.array)
        Return:
            text (str): Text in this image (auto upper)
        '''
        if (self.cfg.PIPELINE.TEXT_RECOGNITION.MORAN == True):
            text = moran_text_recog(self.MORAN_CONFIG, image, self.MORAN_TRANSFORMER, self.MORAN_CONVERTER, self.MORAN)
            
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

    def run_full_pipeline_on_image(self, image=None, image_path=None):
        '''
        This function will run detection & recognition in single image
        Args:
            image (np.array)
            image_path (str)
        Return:
            record (dict): dictionary list of region is text and text of this
                {
                    image_name: name of image
                    result:
                    [
                        {
                            x_min
                            y_min
                            x_max
                            y_max
                            text
                        }
                        .
                        .
                        .
                        {
                            x_min
                            y_min
                            x_max
                            y_max
                            text
                        }

                    ]
                }
        '''        
        if (image_path != None):
            image = cv2.imread(image_path)
            image_name = image_path.split('/')[-1]

        print ('Run detection & recognition on image: ', image_name)
        # record list to save result
        record = {}
        record['image_name'] = image_name
        self.objs = []
        
        # text detect
        start_time = time.time()
        bboxes = self.detection(image)

        #text recognition loop over all text region
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])
            text = self.recognition_on_region_of_image(image, x_min, y_min, x_max, y_max)
            self.append_objs(x_min, y_min, x_max, y_max, text)
        
        end_time = time.time()
        print('Time process {} is {}s'.format(image_name, end_time-start_time))
        # save result to json file
        record['result'] = self.objs
        if (self.args.save_visualize == True):
            self.visualize(image, record)
        return record

    def run_full_pipeline_on_folder(self, folder_path):
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            record = self.run_full_pipeline_on_image(image_path = image_path)
            self.output_dict.append(record)
        
        self.write_file_result()

            
    def visualize(self, image, record=None, bboxes_detection=None, image_name=None):
        '''
        This function have 2 mode to visualize
        1. when record=None and bboxes_detection!=None, it will save detection visualize image
        2. When record!=None and bboxes_detection==None, it will save full pipeline visualize image
        Args: 
            image (array): image for save
            record (dict): a result instances when run end to end with function run full pipeline in folder
                {
                    image_name: name of image
                    result:
                    [
                        {
                            x_min
                            y_min
                            x_max
                            y_max
                            text
                        }
                        .
                        .
                        .
                        {
                            x_min
                            y_min
                            x_max
                            y_max
                            text
                        }

                    ]
                }
            bboxes_detection (array): detection result, is list of array with this form,
            where an array is represent for a bounding box detection
                [[x1, y1, x2, y2, x3, y3, x4, y4], ..., [x1, y1, x2, y2, x3, y3, x4, y4]]
        '''
        if (record != None):
            image_save_path = os.path.join(self.args.save_image_folder, record['image_name'])
        else:
            image_save_path = os.path.join(self.args.save_image_folder, image_name)
        # save detection result visualize
        if (self.args.run_detection_only == True):
            for bbox in bboxes_detection:
                x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
                x3, y3, x4, y4 = bbox[2][0], bbox[2][1], bbox[3][0], bbox[3][1]
                cv2.rectangle(image, (int(x1), int(y1)), (int(x3), int(y3)), (255, 255, 0), 1)
                if (self.args.show_visualize == True):
                    cv2.imshow('result', image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        # save full pipeline result visualize
        else:
            for obj in record['result']:
                x_min = obj['x_min']
                y_min = obj['y_min']
                x_max = obj['x_max']
                y_max = obj['y_max']
                text = obj['text']  
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)
                cv2.putText(image, text, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)
                if (self.args.show_visualize == True):
                    cv2.imshow('result', image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        cv2.imwrite(image_save_path, image)
        
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
    parser.add_argument('--config_pipeline', type=str, default='./configs/pipeline.yaml',
                        help='Path to config pipeline file')
    parser.add_argument('--config_detection', type=str, default='./configs/craft.yaml',
                        help='Path to config detection file')
    parser.add_argument('--config_recognition', type=str, default='./configs/MORANv2.yaml',
                        help='Path to config recognition file')
    parser.add_argument('--run_on_folder', type=bool, default=False,
                        help='Wheter or not run on folder')
    parser.add_argument("-i", "--image_path", type=str,
                        help='If run_on_folder = False it require, pipeline auto run on image')
    parser.add_argument("--folder_path", type=str, default='./data',
                        help='If run_on_folder = True, pipeline will run on this folder_path')
    parser.add_argument("--run_full_pipeline", type=bool, default=False,
                        help='Wheter or not run full pipeline')
    parser.add_argument("--run_detection_only", type=bool, default=False,
                        help='Wheter or not run detection only')
    parser.add_argument("--detection_weight", type=str, default='./models/craft_model.pth',
                        help='Path to detection weight')
    parser.add_argument("--run_recognition_only", type=bool, default=False,
                        help='Wheter or not run detection only')
    parser.add_argument("--recognition_weight", type=str, default='./models/moranv2_model.pth',
                        help='Path to recognition weights')
    parser.add_argument("--run_recognition_with_detection_result", type=bool, default=False,
                        help='Wheter or not run recognition with detection result, if True require detection output path as input folder')
    parser.add_argument("--full_pipeline_output_path", type=str, default='./output_full_pipeline',
                        help='Path to full pipeline output, file will save as json')
    parser.add_argument("--detection_output_path", type=str, default='./output_detection',
                        help='Path to detection output')
    parser.add_argument("--recognition_output_path", type=str, default='./output_recognition',
                        help='Path to recognition output')
    parser.add_argument("-sv", "--save_visualize", type=bool, default=True,
                        help='Wheter or not save visualize recognition')
    parser.add_argument("--show_visualize", type=bool, default=False,
                        help='Wheter or not show visualize image after running')
    parser.add_argument('--save_image_folder', type=str, default='./visualize_output',
                        help='Path to output visualize')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # setup config
    cfg = get_config()
    cfg.merge_from_file('configs/pipeline.yaml')
    cfg.merge_from_file('configs/craft.yaml')
    cfg.merge_from_file('configs/MORANv2.yaml')

    # create output folder
    if (os.path.exists(args.full_pipeline_output_path) == False):
        os.mkdir(args.full_pipeline_output_path)
    if (os.path.exists(args.detection_output_path) == False):
        os.mkdir(args.detection_output_path)
    if (os.path.exists(args.recognition_output_path) == False):
        os.mkdir(args.recognition_output_path)
    if(args.save_visualize == True and os.path.exists(args.save_image_folder) == False):
        os.mkdir(args.save_image_folder)
             
    image_recognition = ImageRecognition(cfg, args)
    
    if (args.run_on_folder == True):
        if (args.run_full_pipeline == True):
            image_recognition.run_full_pipeline_on_folder(args.folder_path)
        if (args.run_detection_only == True):
            image_recognition.run_detection_on_folder(args.folder_path)
        if (args.run_recognition == True):
            image_recognition.run_recognition_on_folder(args.folder_path)

    else:
        if (args.image_path == None):
            print ("Can't find path to image, please use --image_path arguments")
            exit()
        else:
            if (args.run_full_pipeline == True):
                if (args.run_detection_only == True or args.run_recognition_only == True):
                    print ("Please turn off detection only mode or recognition only mode by args")
                    exit()
                else:
                    image_recognition.run_full_pipeline_on_image(image_path=args.image_path)
            if (args.run_detection_only == True):
                if (args.run_full_pipeline == True or args.run_recognition_only == True):
                    print ("Please turn off run full pipeline mode or recognition only mode by args")
                    exit()
                else:
                    image_recognition.detection(image_path=args.image_path)
            if (args.run_recognition == True):
                if (args.run_full_pipeline == True or args.run_detection_only == True):
                    print ("Please turn off run full pipeline mode or detection only mode by args")
                    exit()
                else:
                    image_recognition.recognition(image_path=args.image_path)


    