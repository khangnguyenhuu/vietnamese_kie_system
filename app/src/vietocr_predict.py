import sys
sys.path.insert(0, "../../")
from libs.vietocr.vietocr.tool.predictor import Predictor
from libs.vietocr.vietocr.tool.config import Cfg
from .base import TextRecog
class VietOCR(TextRecog):

    def __init__(self, config_path="./experiments/vietocr/config.yml"):
        self.config = Cfg.load_config_from_file(config_path)
        self.model = Predictor(self.config)
        
    def recognize(self, img):
        text = self.model.predict(img)
        return text


