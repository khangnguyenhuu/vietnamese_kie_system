import torch
from torch.autograd import Variable
from libs.MORAN_v2.tools import utils 
from libs.MORAN_v2.tools import dataset 
from PIL import Image
from collections import OrderedDict
import cv2
from libs.MORAN_v2.models.moran import MORAN
from src.base import TextRecog

class myMoran(TextRecog):
    def __init__(self, config, model_weight_path, MORAN=MORAN):
        self.cfg = config
        self.model_weight_path = model_weight_path
        self.MORAN = MORAN
        self.load_model_moran()
        

    def load_model_moran(self):
        
        if torch.cuda.is_available():
            self.cfg.CUDA = True
            self.MORAN = self.MORAN(1, len(self.cfg.ALPHABET.split(':')), 256, 32, 100, BidirDecoder=True, CUDA=self.cfg.CUDA)
            self.MORAN = self.MORAN.cuda()
        else:
            self.MORAN = self.MORAN(1, len(self.cfg.ALPHABET.split(':')), 256, 32, 100, BidirDecoder=True, inputDataType='torch.FloatTensor', CUDA=self.cfg.CUDA)

        if self.cfg.CUDA:
            state_dict = torch.load(self.model_weight_path)
        else:
            state_dict = torch.load(self.model_weight_path, map_location='cpu')
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "") # remove `module.`
            MORAN_state_dict_rename[name] = v
        self.MORAN.load_state_dict(MORAN_state_dict_rename)

        for p in self.MORAN.parameters():
            p.requires_grad = False
        self.MORAN.eval()

        self.converter = utils.strLabelConverterForAttention(self.cfg.ALPHABET, ':')
        self.transformer = dataset.resizeNormalize((100, 32))
    
    def recognize(self, image):
        h, w, c = image.shape
        if c == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = Image.fromarray(image)
        image = self.transformer(image)
        
        if self.cfg.CUDA:
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)
        text = torch.LongTensor(1 * 5)
        length = torch.IntTensor(1)
        text = Variable(text)
        length = Variable(length)

        max_iter = 20
        t, l = self.converter.encode('0'*max_iter)
        utils.loadData(text, t)
        utils.loadData(length, l)
        output = self.MORAN(image, length, text, text, test=True, debug=True)

        preds, preds_reverse = output[0]
        demo = output[1]

        _, preds = preds.max(1)
        _, preds_reverse = preds_reverse.max(1)

        sim_preds = self.converter.decode(preds.data, length.data)
        sim_preds = sim_preds.strip().split('$')[0]
        return sim_preds
