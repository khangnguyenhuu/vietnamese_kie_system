#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import cv2
import copy
import os.path as osp
import warnings

import mmcv
import torch

from mmcv.utils.config import Config
from mmcv.runner import load_checkpoint
from mmdet.core import encode_mask_results
from mmocr.utils.fileio import list_from_file
from mmocr.models import build_detector
from mmocr.apis.inference import model_inference
from mmocr.datasets.kie_dataset import KIEDataset
from mmocr.utils import revert_sync_batchnorm

from .base import KIE
class SDMGR():
    def __init__(self, \
        		    config_path="./app/experiments/sdmgr/sdmgr_unet16_60e_wildreceipt.py", \
                model_path="./app/experiments/sdmgr/epoch_54.pth", \
                device="cuda:0"):
        # build the model and load checkpoint
        self.model = None
        cfg = Config.fromfile(config_path)
        self.model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        self.model = revert_sync_batchnorm(self.model)
        self.model.cfg = cfg
        load_checkpoint(self.model, model_path, map_location=device)
        self.kie_dataset = KIEDataset(dict_file=self.model.cfg.data.test.dict_file)
    
    def generate_kie_labels(self, result, boxes, class_list):
        idx_to_cls = {}
        if class_list is not None:
            for line in list_from_file(class_list):
                class_idx, class_label = line.strip().split()
                idx_to_cls[class_idx] = class_label

        max_value, max_idx = torch.max(result['nodes'].detach().cpu(), -1)
        node_pred_label = max_idx.numpy().tolist()
        node_pred_score = max_value.numpy().tolist()
        labels = []
        for i in range(len(boxes)):
            pred_label = str(node_pred_label[i])
            text = boxes[i]
            if pred_label in idx_to_cls:
                pred_label = idx_to_cls[pred_label]
            # pred_score = node_pred_score[i]
            labels.append([pred_label, text])
        return labels
    
    def extract_infor(self, image, bboxes_text):
        '''
        bboxes_text:
            1, 1, 1, 1, 1, 1, 1, 1, text
        '''
        print(bboxes_text)
        img_e2e_res = {}
        img_e2e_res['filename'] = "temp"
        img_e2e_res['result'] = []
        box_imgs = []
        for bbox in bboxes_text:
            box_res = {}
            box_res['box'] = [round(x) for x in bbox[:-1]]
            box_res['box_score'] = 1
            box_res['text'] = bbox[-1]
            box_res['text_score'] = 1
            img_e2e_res['result'].append(box_res)
        
        annotations = copy.deepcopy(img_e2e_res['result'])
        # Customized for kie_dataset, which
        # assumes that boxes are represented by only 4 points
        for i, ann in enumerate(annotations):
            min_x = min(ann['box'][::2])
            min_y = min(ann['box'][1::2])
            max_x = max(ann['box'][::2])
            max_y = max(ann['box'][1::2])
            annotations[i]['box'] = [
                min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
            ]
        ann_info = self.kie_dataset._parse_anno_info(annotations)
        ann_info['ori_bboxes'] = ann_info.get('ori_bboxes',
                                                ann_info['bboxes'])
        ann_info['gt_bboxes'] = ann_info.get('gt_bboxes',
                                                ann_info['bboxes'])
        kie_result, data = model_inference(self.model, \
                                            image, \
                                            ann=ann_info, \
                                            return_data=True)
        kie_result = self.generate_kie_labels(kie_result, bboxes_text, "./data_kie/class_list.txt")
        return kie_result

