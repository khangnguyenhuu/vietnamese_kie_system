from src.deeptext.deeptext_predict import MyDeepText
import numpy as np
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from libs.deeptext.utils import CTCLabelConverter, AttnLabelConverter
from libs.deeptext.dataset import RawDataset, AlignCollate
from libs.deeptext.model import Model

from PIL import Image

from src.base import TextRecog

from utils.parser import get_config
if __name__ == '__main__':
    opt = get_config('configs/recognize_text/deeptext.yaml')
    model = myDeepText(opt)
    print(model.recognize(Image.open('./data/a.png')))