
import os

import cv2
import torch
from collections import OrderedDict
import torch.backends.cudnn as cudnn

from libs.CRAFT.predict import test_net
from libs.CRAFT import file_utils

import os 
import time

import cv2
import torch
import argparse
import numpy as np
import json

from utils import get_config, loadImage, tlwh_2_maxmin

from libs.CRAFT.craft import CRAFT
from src.base import TextDetector

class MyCRAFT(TextDetector):

    def __init__(self,config):
        self.config = config
        self.model = CRAFT()
        self.refine_net = None
        self.load_model_craft()

    def load_model_craft(self):
        print(self.config)
        print('Loading weights CRAFT from checkpoint (' + self.config.TRAINED_MODEL + ')')
        if self.config.CUDA:
            self.model.load_state_dict(self.copyStateDict(torch.load(self.config.TRAINED_MODEL)))
        else:
            self.model.load_state_dict(self.copyStateDict(torch.load(self.config.TRAINED_MODEL, map_location='cpu')))

        if self.config.CUDA:
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = False

        self.model.eval()
        # LinkRefiner
        
        if self.config.REFINE:
            from libs.CRAFT.refinenet import RefineNet
            self.refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + self.config.refiner_model + ')')
            if self.config.CUDA:
                self.refine_net.load_state_dict(copyStateDict(torch.load(self.config.refiner_model)))
                self.refine_net = self.refine_net.cuda()
                self.refine_net = torch.nn.DataParallel(self.refine_net)
            else:
                self.refine_net.load_state_dict(copyStateDict(torch.load(self.config.refiner_model, map_location='cpu')))

            self.refine_net.eval()
            self.config.POLY = True
        # return self.model

    def copyStateDict(self,state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    def detect(self,image):    
        
        bboxes, _, _ = test_net(self.model, image, self.config.TEXT_THRESHOLD, self.config.LINK_THRESHOLD, self.config.LOW_TEST, self.config.CUDA, self.config.POLY, self.refine_net, self.config)

        return  bboxes
