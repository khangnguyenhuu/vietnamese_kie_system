
import numpy
import json
def cal_tp(preds_path, gt_path):
    '''
            preds:
            {
                "abc.jpg": {
                    "result": [
                        [2, 3, 4, 5, 6, 7, 8, 9, "abcds", "Ngay_gui"], 
                        [2, 3, 4, 5, 6, 7, 8, 9, "abcds", "Ngay_gui"]
                    
                    ]},
                "def.jpg": {
                    "result": [
                        [2, 3, 4, 5, 6, 7, 8, 9, "def", "Ngay_gui"], 
                        [2, 3, 4, 5, 6, 7, 8, 9, "def", "Ngay_gui"]
                        
                ]}
                
            }
        ground_truths:
            {
                "abc.jpg": {
                    "result": [
                        [2, 3, 4, 5, 6, 7, 8, 9, "abcds", "Ngay_gui"], 
                        [2, 3, 4, 5, 6, 7, 8, 9, "abcds", "Ngay_gui"]
                    
                    ]},
                "def.jpg": {
                    "result": [
                        [2, 3, 4, 5, 6, 7, 8, 9, "def", "Ngay_gui"], 
                        [2, 3, 4, 5, 6, 7, 8, 9, "def", "Ngay_gui"]
                        
                ]}
                
            }
    args:
        preds_path
        gt_path
    return:
        f1_score_per_class:
        avg_f1_score:
    '''
    class_dict = {
        "So_Ke_hoach": {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
        },
        "Ngay_gui": {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
        },
        "So": {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
        },
        "Noi_nhan": {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
        },
        "Nguoi_ky": {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
        },
        "Ngay_ke_hoach": {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
        },
    }
    gt_dict = {
    "So_Ke_hoach": "",
    "Ngay_gui": "",
    "So": "",
    "Noi_nhan": "",
    "Nguoi_ky": "",
    "Ngay_ke_hoach": ""

    }   
    with open(preds_path, encoding="utf-8") as fp1:
        with open(gt_path, encoding="utf-8") as fp2:
            preds = json.load(fp1)
            gts = json.load(fp2)
            for gt in gts:
                gt = gts[gt]["result"]
                So_Ke_hoach = []
                Ngay_gui = []
                So = []
                Noi_nhan = []
                Nguoi_ky = []
                Ngay_ke_hoach = []
                for row in gt:
                    if row[-1] == "So_Ke_hoach":
                        So_Ke_hoach.append(row)
                    if row[-1] == "Ngay_gui":
                        Ngay_gui.append(row)
                    if row[-1] == "So":
                        So.append(row)    
                    if row[-1] == "Noi_nhan":
                        Noi_nhan.append(row)
                    if row[-1] == "Nguoi_ky":
                        Nguoi_ky.append(row)
                    if row[-1] == "Ngay_ke_hoach":
                        Ngay_ke_hoach.append(row)
                gt_dict.update({"So_Ke_hoach": So_Ke_hoach})
                gt_dict.update({"Ngay_gui": Ngay_gui})
                gt_dict.update({"So": So})
                gt_dict.update({"Noi_nhan": Noi_nhan})
                gt_dict.update({"Nguoi_ky": Nguoi_ky})
                gt_dict.update({"Ngay_ke_hoach": Ngay_ke_hoach})
            for pred in preds:
                pred = preds[pred]["result"]
                for row in pred:
                    print(row)
                    class_name = row[-1]
                    text = row[-2]
                    for field in class_dict:
                        field = field.replace(":", "")
                        if class_name == field:
                            if row in gt_dict[field]:
                                class_dict[field]['tp'] += 1
                            else:
                                class_dict[field]['fp'] += 1
                        if class_name != field:
                            if row in gt_dict[field]:
                                class_dict[field]['fn'] += 1
                            else:
                                class_dict[field]['tn'] += 1
    for field in class_dict:
        precision = class_dict[field]['tp']/(class_dict[field]['tp']+class_dict[field]['fp']) 
        recall = class_dict[field]['tp']/(class_dict[field]['tp']+class_dict[field]['fn'])
        f1 = 2*precision*recall/(precision+recall)
        field["precision"] = precision
        field["recall"] = recall
        field["f1"] = f1
    print(class_dict)

cal_tp("./preds.json", "./gt.json")