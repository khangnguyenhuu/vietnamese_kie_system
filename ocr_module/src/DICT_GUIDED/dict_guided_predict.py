'''
author: Khangnh
Last modify: 07/07/2021
'''

import argparse
import glob
import multiprocessing as mp
import os
import time

import cv2
import tqdm
from libs.dict_guided.adet.config import get_cfg as get_cfg_guided_text
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from libs.dict_guided.demo.predictor import VisualizationDemo

from utils import get_config, loadImage, tlwh_2_maxmin


def setup_cfg(cfg_guided, args):
    # load config from file and command-line arguments
    cfg = get_cfg_guided_text()
    cfg.merge_from_file(cfg_guided.DICT_GUIDED.CONFIG)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = cfg_guided.DICT_GUIDED.CONFIDENCE
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg_guided.DICT_GUIDED.CONFIDENCE
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = cfg_guided.DICT_GUIDED.CONFIDENCE
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = cfg_guided.DICT_GUIDED.CONFIDENCE
    cfg.MODEL.DEVICE= cfg_guided.DICT_GUIDED.DEVICE
    cfg.freeze()
    return cfg

def load_model_dict_guided(cfg, args):
    mp.set_start_method("spawn", force=True)
    cfg_detectron2 = setup_cfg(cfg, args)
    predictor = VisualizationDemo(cfg_detectron2)
    return predictor

def dict_guided_predict(predictor, image_path):
    img = read_image(image_path, format="BGR")
    prediction, visualized_output = predictor.run_on_image(img)
    return prediction, visualized_output