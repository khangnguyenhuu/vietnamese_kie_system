import sys
sys.append("../", 0)
from libs.vietocr.tool.predictor import Predictor
from libs.vietocr.tool.config import Cfg

class VietOCR(TextRecog):

    def __init__(self, config_path="./experiments/vietocr/config.yml"):
        self.config = Cfg.load_config_from_file("model/vietocr/config.yml")
        self.model = Predictor(config_reg)
        
    def recognize(self, img):
        text = self.model.predict(img)
        return text


