import os
import time
import yaml
import argparse
from PIL import Image
import matplotlib.pyplot as plt

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="foo help")
    # parser.add_argument("--folder_img", required=True, help="foo help")
    parser.add_argument("--config", required=True, help="foo help")

    args = parser.parse_args()
    
    config = Cfg.load_config_from_file(args.config)
    config[
        "vocab"
    ] = ''' !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\xB0\
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
\u2026\u20AC\u2122\u2212'''
    print(config)

    detector = Predictor(config)
    f_pre = open("./vietocr_result.txt", "w+")

    # Option for predicting folder images
    '''img_list = os.listdir(args.img)
    img_list = sorted(img_list)

    start_time = time.time()
    for img in img_list:
        img_path = args.img + img
        image = Image.open(img_path)

        s = detector.predict(image)
        print(img_path, "-----", s)

        res = img + "\t" + s + "\n"
        f_pre.write(res)
    runtime = time.time() - start_time
    print("FPS:", len(img_list) / runtime)'''

    # Option for predicting folder images
    start_time = time.time()
    img_path = args.img
    image = Image.open(img_path)

    s = detector.predict(image)
    print(img_path, "-----", s)

    res = args.img + "\t" + s + "\n"
    f_pre.write(res)
    runtime = time.time() - start_time    


if __name__ == "__main__":
    main()
