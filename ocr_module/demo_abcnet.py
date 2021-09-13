'''
author: pxtri
last modified: 06/07/2021
'''

import argparse
import glob
import multiprocessing as mp
import os
import time

import cv2
import tqdm
from libs.ABCnet.adet.config import get_cfg as get_cfg_abcnet
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from libs.ABCnet.demo.predictor import VisualizationDemo

from src.ABCnet.abcnet_predict import load_model_abcnet, abcnet_predict
from utils import get_config, loadImage, tlwh_2_maxmin


def setup_cfg(cfg_abcnet, args):
    # load config from file and command-line arguments
    cfg = get_cfg_abcnet()
    cfg.merge_from_file(cfg_abcnet.CONFIG)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = cfg_abcnet.CONFIDENCE
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg_abcnet.CONFIDENCE
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = cfg_abcnet.CONFIDENCE
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = cfg_abcnet.CONFIDENCE
    cfg.freeze()
    return cfg

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_on_folder', type=bool, default=False,
                        help='Wheter or not run on folder')
    parser.add_argument("-i", "--image_path", type=str,
                        help='Path to image')
    parser.add_argument("--folder_path", type=str, default='./data',
                        help='Path to folder')
    parser.add_argument("-sv", "--save_visualize", type=bool, default=True,
                        help='Wheter or not save visualize image')
    parser.add_argument('--save_image_folder', type=str, default='./visualize_output',
                        help='Path to save visualize image')
    parser.add_argument(
    "--opts",
    help="Argument of Dict guided method, modify config options using the command-line 'KEY VALUE' pairs",
    default=[],
    nargs=argparse.REMAINDER,
    )
    return parser.parse_args()

if __name__=="__main__":

    # set up args & config
    cfg = get_config()
    cfg.merge_from_file("./configs/abcnet.yaml")
    args = parse_args()

    # loading model
    print ('[LOADING] MODEL ABCnet')
    predictor = load_model_abcnet(cfg.ABC, args)
    print ('[LOADING SUCESS] MODEL ABCnet')

    if (args.save_visualize == True and os.path.exists(args.save_image_folder) == False):
        os.mkdir(args.save_image_folder)

    # predict
    # run on folder
    if (args.run_on_folder == True):
        print ("Run on folder ", args.folder_path)
        for img_name in os.listdir(args.folder_path):
            image_path = os.path.join(args.folder_path, img_name)
            print("image path: ",image_path )
            start = time.time()
            prediction, visualize_output = abcnet_predict(predictor, image_path)
            end = time.time()
            print ('Run predict on image {} take {}s'.format(img_name, (end-start)))
            if (args.save_visualize == True):
                print ("Visualize image will save at {}".format(args.save_image_folder))
                output_path = os.path.join(args.save_image_folder, img_name)
                visualize_output.save(output_path)

    # run on single image
    else:
        print ("Run on image: ", args.image_path)
        prediction, visualize_output = abcnet_predict(predictor, args.image_path)
        if (args.save_visualize == True):
            print ("Visualize image will save at {}".format(args.save_image_folder))
            output_path = os.path.join(args.save_image_folder, args.image_path)
            visualize_output.save(output_path)



'''
Run code:
CUDA_VISIBLE_DEVICES=2 python demo_abcnet.py \
    --run_on_folder True \
    --folder_path /mlcv/WorkingSpace/SceneText/tripx/vietscenetext_framework/libs/ABCnet/test_image \
    --save_visualize ./libs/ABCnet/output_test \
    --opts MODEL.WEIGHTS ./libs/ABCnet/models/ctw1500_attn_R_50.pth

Download model: 
 wget -O ctw1500_attn_R_50.pth https://universityofadelaide.box.com/shared/static/okeo5pvul5v5rxqh4yg8pcf805tzj2no.pth
'''