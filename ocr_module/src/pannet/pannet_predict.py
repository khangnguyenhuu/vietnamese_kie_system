from libs.MMOCR.mmocr.utils.ocr import MMOCR
from libs.MMOCR.mmocr.apis import init_detector
from libs.MMOCR.mmocr.utils.model import revert_sync_batchnorm
from utils.utils import convert_xyminmax
from ..base import TextDetector

class PanNet(TextDetector):
	
	def __init__(self, \
				config_path="libs/MMOCR/configs/textdet/panet/panet_r18_fpem_ffm_600e_ctw1500.py", \
				model_path="experiments/panet/20ep_data_16point.pth", \
				device="cuda:0"):
		self.ocr = MMOCR()
		self.device = device
		self.config_path = config_path
		self.model_path = model_path
		self.detect_model = init_detector(self.config_path, self.model_path, device=self.device)
		self.detect_model = revert_sync_batchnorm(self.detect_model)

	def detect(self, img):
		list_box_results = self.ocr.readtext(img, detect_model=self.detect_model)
		list_box_results = convert_xyminmax(list_box_results)
		return list_box_results