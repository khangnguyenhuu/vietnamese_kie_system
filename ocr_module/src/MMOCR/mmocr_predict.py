from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401


from src.base_detect import TextDetector


class MyMMOcr(TextDetector):
	def __init__(self,config):
		self.config = config
		self.model = init_detector(self.config.CONFIG_PY, self.config.CHECKPOINT)
	def detect(self,image): 
		result = model_inference(self.model , image)
		result_bb = result['boundary_result']
		return result_bb