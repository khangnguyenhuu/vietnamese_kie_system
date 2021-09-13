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


class myDeepText(TextRecog):
    def __init__(self, opt, model_weight_path):
        opt.character = "AàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!\"#$%&''()*+,-./:;<=>?@[\]^_`{|}~ "
        opt.num_class = len(opt.character)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if 'CTC' in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        
        if opt.rgb:
            opt.input_channel = 3
        model = Model(opt)

        model = torch.nn.DataParallel(model).to(device)

        # Load model
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        self.model = model
        self.opt = opt
        self.device = device
        self.converter = converter


    def recognize(self, np_image, return_prob=False):
        image = Image.fromarray(np.uint8(np_image)).convert('RGB')
        image.save('libs/deeptext/images/im.jpg')
        self.opt.image_folder = 'libs/deeptext/images'
        AlignCollate_image = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)
        image_data = RawDataset(root=self.opt.image_folder, opt=self.opt)  # use RawDataset
        image_loader = torch.utils.data.DataLoader(
            image_data, batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=AlignCollate_image, pin_memory=True)
        
        # predict
        self.model.eval()
        with torch.no_grad():
            for image_tensors, image_path_list in image_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(self.device)
                # For max length prediction
                length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(self.device)
                text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)

                if 'CTC' in self.opt.Prediction:
                    preds = self.model(image, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = self.converter.decode(preds_index, preds_size)

                else:
                    preds = self.model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, length_for_pred)

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    if 'Attn' in self.opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    if return_prob:
                        return pred, confidence_score
                    return pred


if __name__ == '__main__':
    opt = get_config('configs/deeptext.yaml')
    model = DeepText(opt)
    print(model.recognize(Image.open('../../../demo_images/demo.jpg')))
