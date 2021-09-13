'''
author: pxtri
Last modify: 06/07/2021
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
    cfg.MODEL.DEVICE="cpu"
    cfg.freeze()
    return cfg

def load_model_abcnet(cfg, args):
    mp.set_start_method("spawn", force=True)
    cfg_detectron2 = setup_cfg(cfg, args)
    predictor = VisualizationDemo(cfg_detectron2)
    return predictor

def abcnet_predict(predictor, image_path):
    img = read_image(image_path, format="BGR")
    prediction, visualized_output = predictor.run_on_image(img)
    return prediction, visualized_output