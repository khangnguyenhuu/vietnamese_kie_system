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

class SDMGR():
    def __init__(self, \
        		config_path="./work_dirs/sdmgr_unet16_60e_wildreceipt/sdmgr_unet16_60e_wildreceipt.py", \
                model_path="./work_dirs/sdmgr_unet16_60e_wildreceipt/latest.pth", \
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

model = SDMGR()
js = {"annotations": [{"box": [221.0, 87.0, 702.0, 86.0, 702.0, 120.0, 222.0, 121.0], "text": "BÔ KHOA HOC VÀ CÔNG NGHỆ", "label": 5}, {"box": [874.0, 84.0, 1563.0, 81.0, 1563.0, 117.0, 874.0, 120.0], "text": "CỘNG HOÀ XÃ HÔI CHỦ NGHĨA VIÊT NAM", "label": 5}, {"box": [289.0, 136.0, 634.0, 133.0, 635.0, 170.0, 289.0, 172.0], "text": "CỰC SỞ HỮU TRÍ TUÊ", "label": 5}, {"box": [1015.0, 135.0, 1417.0, 135.0, 1417.0, 167.0, 1015.0, 167.0], "text": "Độc lập - Tự do - Hạnh phúc", "label": 5}, {"box": [853.0, 216.0, 1337.0, 213.0, 1337.0, 249.0, 853.0, 252.0], "text": "Hà Nội ngày 20 tháng 01 năm 2017", "label": 1}, {"box": [336.0, 234.0, 588.0, 236.0, 588.0, 266.0, 336.0, 264.0], "text": "S6:4093/QĐ-SHTT", "label": 0}, {"box": [771.0, 368.0, 998.0, 364.0, 999.0, 407.0, 771.0, 411.0], "text": "QUYẾT ĐỊNH", "label": 5}, {"box": [665.0, 417.0, 1108.0, 423.0, 1107.0, 458.0, 664.0, 453.0], "text": "Về việc chấp nhận đơn hợp lệ", "label": 5}, {"box": [570.0, 486.0, 1198.0, 481.0, 1198.0, 523.0, 571.0, 528.0], "text": "CỤC TRƯỞNG CỤC SỞ HỮU TRÍ TUỆ", "label": 5}, {"box": [235.0, 562.0, 1422.0, 560.0, 1422.0, 596.0, 235.0, 598.0], "text": "Căn cứ Điều lệ Tổ chức và Hoạt động của Cục Sở hữu trí tuệ ban hành theo Quyết định số", "label": 5}, {"box": [187.0, 612.0, 1204.0, 612.0, 1204.0, 645.0, 187.0, 645.0], "text": "69/QĐ-BKHCN ngày 15/01/2014 của Bộ trưởng Bộ Khoa học và Công nghệ;", "label": 5}, {"box": [239.0, 665.0, 1511.0, 664.0, 1511.0, 693.0, 239.0, 694.0], "text": "Căn cứ điểm 13.2 và điểm 13.6.b của Thông tư số 01/2007/TT-BKHCN ngày 14.02.2007 của Bộ", "label": 5}, {"box": [188.0, 714.0, 1197.0, 710.0, 1197.0, 743.0, 189.0, 747.0], "text": "Khoa học và Công nghệ hướng dẫn thi hành Nghị định số 103/2006/NĐ-CP:", "label": 5}, {"box": [240.0, 764.0, 1124.0, 763.0, 1124.0, 793.0, 240.0, 794.0], "text": "Cãn cứ kết quả thẩm đinh hình thức đơn đăng ký giải pháp hữu ích", "label": 5}, {"box": [291.0, 814.0, 582.0, 814.0, 582.0, 841.0, 291.0, 841.0], "text": "Số đơn: 2-2016-00434", "label": 5}, {"box": [791.0, 876.0, 1024.0, 872.0, 1025.0, 915.0, 791.0, 920.0], "text": "QUYẾT ĐỊNH", "label": 5}, {"box": [238.0, 938.0, 1016.0, 936.0, 1016.0, 971.0, 238.0, 973.0], "text": "Điều 1 Chấp nhận đơn hợp lệ với những ghi nhận sau đây:", "label": 5}, {"box": [291.0, 987.0, 643.0, 983.0, 644.0, 1013.0, 291.0, 1017.0], "text": "Ngày nộp đơn: 09/12/2016", "label": 5}, {"box": [290.0, 1027.0, 1036.0, 1025.0, 1036.0, 1058.0, 290.0, 1060.0], "text": "Chủ đơn(): Trường Đại học Công nghệ Thông tin (VN)", "label": 5}, {"box": [289.0, 1075.0, 1351.0, 1072.0, 1351.0, 1105.0, 289.0, 1108.0], "text": "Địa chỉ: Khu phố 6 phường Linh Trung quận Thủ Đức thành phố Hồ Chí Minh", "label": 5}, {"box": [289.0, 1123.0, 1475.0, 1121.0, 1475.0, 1156.0, 289.0, 1158.0], "text": "Tên giải pháp hữu ích: Phương pháp phát hiện biển báo giao thông sử dụng kết hợp đa đặc", "label": 5}, {"box": [288.0, 1171.0, 361.0, 1176.0, 358.0, 1208.0, 286.0, 1203.0], "text": "trưng", "label": 5}, {"box": [239.0, 1220.0, 1520.0, 1220.0, 1520.0, 1252.0, 239.0, 1252.0], "text": "Điều 2 Công bố đơn trên Công báo sở hữu công nghiệp và thẩm định nội dung trong trường hợp", "label": 5}, {"box": [236.0, 1266.0, 1265.0, 1261.0, 1265.0, 1297.0, 237.0, 1302.0], "text": "có yêu cầu theo quy định tại điểm 25.1 của Thông tư số 01/2007/TT-BKHCN", "label": 5}, {"box": [236.0, 1310.0, 1527.0, 1309.0, 1527.0, 1346.0, 236.0, 1347.0], "text": "Điều 3 Chánh Văn phòng Trưởng phòng Đăng ký Trưởng phòng Thông tin chịu trách nhiệm thi", "label": 5}, {"box": [239.0, 1363.0, 538.0, 1363.0, 538.0, 1390.0, 239.0, 1390.0], "text": "hành Quyết định này./", "label": 5}, {"box": [962.0, 1402.0, 1220.0, 1399.0, 1220.0, 1431.0, 963.0, 1434.0], "text": "TL. CỤC TRƯỞNG", "label": 4}, {"box": [121.0, 1454.0, 250.0, 1454.0, 250.0, 1481.0, 121.0, 1481.0], "text": "Nơi nhân:", "label": 5}, {"box": [808.0, 1448.0, 1372.0, 1441.0, 1372.0, 1476.0, 808.0, 1484.0], "text": "PHÓ TRƯỞNG PHÒNG SÁNG CHẾ SỐ 1", "label": 4}, {"box": [120.0, 1496.0, 518.0, 1497.0, 518.0, 1523.0, 120.0, 1522.0], "text": "- Chủ đơn/ đai diên của chủ đơn:", "label": 3}, {"box": [121.0, 1539.0, 304.0, 1541.0, 304.0, 1567.0, 121.0, 1565.0], "text": "- Lưu: VT HT", "label": 3}, {"box": [107.0, 2149.0, 1008.0, 2146.0, 1008.0, 2169.0, 107.0, 2172.0], "text": "(*)Trong trường hợp đơn có nhiều chủ đơn đây là chủ đơn thứ nhất ghi trong danh sách các chủ đơn", "label": 5}, {"box": [120.0, 299.0, 401.0, 299.0, 401.0, 333.0, 120.0, 333.0], "text": "TRƯỜNG ĐẠI HỌC", "label": 5}, {"box": [447.0, 337.0, 81.0, 337.0, 81.0, 371.0, 447.0, 371.0], "text": "CÔNG NGHỆ THÔNG TIN", "label": 5}, {"box": [76.0, 415.0, 165.0, 415.0, 165.0, 481.0, 76.0, 481.0], "text": "ĐẾN", "label": 5}, {"box": [233.0, 386.0, 187.0, 386.0, 187.0, 431.0, 233.0, 431.0], "text": "Số ", "label": 2}, {"box": [446.0, 433.0, 186.0, 433.0, 186.0, 475.0, 446.0, 475.0], "text": "Ngày 06.02.2017", "label": 1}, {"box": [968.0, 1663.0, 1212.0, 1663.0, 1212.0, 1696.0, 968.0, 1696.0], "text": "Phan Thanh Hải", "label": 4}, {"box": [247.0, 381.0, 322.0, 381.0, 322.0, 431.0, 247.0, 431.0], "text": "73", "label": 2}]}
box_text = []
for line in js["annotations"]:
    # print(type(line))
    result_box = []
    box = line["box"]
    for i in box:
        result_box.append(i)
    result_box.append(line["text"])
    box_text.append(result_box)
print(box_text)
image = cv2.imread("./data_kie/image_file/0_PVPwIDcHCoL_Jau1fSr3rwlV5W1Vl3aX7xyoK7F14-1.png")
kie_result = model.extract_infor(image, box_text)
print(kie_result)
