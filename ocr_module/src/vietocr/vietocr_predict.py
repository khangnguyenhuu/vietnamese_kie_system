from typing import Text
from libs.vietocr.tool.translate import build_model, translate, translate_beam_search, process_input, predict
from libs.vietocr.tool.utils import download_weights
from src.base import TextRecog

import torch

class myVietOCR(TextRecog):
    def __init__(self, config):
        config[
        "vocab"
    ] = """ !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\xB0\
        \xB2\xC0\xC1\xC2\xC3\xC8\xC9\xCA\xCC\xCD\xD0\xD2\xD3\xD4\xD5\xD6\xD9\xDA\xDC\xDD\
        \xE0\xE1\xE2\xE3\xE8\xE9\xEA\xEC\xED\xF0\xF2\xF3\xF4\xF5\xF6\xF9\xFA\xFC\xFD\u0100\
        \u0101\u0102\u0103\u0110\u0111\u0128\u0129\u014C\u014D\u0168\u0169\u016A\u016B\u01A0\
        \u01A1\u01AF\u01B0\u1EA0\u1EA1\u1EA2\u1EA3\u1EA4\u1EA5\u1EA6\u1EA7\u1EA8\u1EA9\u1EAA\
        \u1EAB\u1EAC\u1EAD\u1EAE\u1EAF\u1EB0\u1EB1\u1EB2\u1EB3\u1EB4\u1EB5\u1EB6\u1EB7\u1EB8\
        \u1EB9\u1EBA\u1EBB\u1EBC\u1EBD\u1EBE\u1EBF\u1EC0\u1EC1\u1EC2\u1EC3\u1EC4\u1EC5\u1EC6\
        \u1EC7\u1EC8\u1EC9\u1ECA\u1ECB\u1ECC\u1ECD\u1ECE\u1ECF\u1ED0\u1ED1\u1ED2\u1ED3\u1ED4\
        \u1ED5\u1ED6\u1ED7\u1ED8\u1ED9\u1EDA\u1EDB\u1EDC\u1EDD\u1EDE\u1EDF\u1EE0\u1EE1\u1EE2\
        \u1EE3\u1EE4\u1EE5\u1EE6\u1EE7\u1EE8\u1EE9\u1EEA\u1EEB\u1EEC\u1EED\u1EEE\u1EEF\u1EF0\
        \u1EF1\u1EF2\u1EF3\u1EF4\u1EF5\u1EF6\u1EF7\u1EF8\u1EF9\u2013\u2014\u2019\u201C\u201D\
        \u2026\u20AC\u2122\u2212"""
        device = config['device']
        
        model, vocab = build_model(config)
        weights = '/tmp/weights.pth'

        if config['weights'].startswith('http'):
            weights = download_weights(config['weights'])
        else:
            weights = config['weights']

        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

        self.config = config
        self.model = model
        self.vocab = vocab
        
    def recognize(self, img, return_prob=False):
        img = process_input(img, self.config['dataset']['image_height'], 
                self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])        
        img = img.to(self.config['device'])

        if self.config['predictor']['beamsearch']:
            sent = translate_beam_search(img, self.model)
            s = sent
            prob = None
        else:
            s, prob = translate(img, self.model)
            s = s[0].tolist()
            prob = prob[0]

        s = self.vocab.decode(s)
        
        if return_prob:
            return s, prob
        else:
            return s
