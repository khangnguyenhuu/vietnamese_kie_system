from libs.AdelaiDet.adet.config import get_cfg
from libs.AdelaiDet.demo.predictor import VisualizationDemo
from utils import get_config
import torch
import numpy as np
from utils.logger import logger

class ABCNetv2:
	
	def __init__(self, args_file):
		logger.info('Initialize ABCNetv2 Detector')
		args = get_config(args_file)
		cfg = self.setup_cfg(args)
		cfg.merge_from_list(["MODEL.WEIGHTS", args.weights])
		self.args = args
		self.cfg = cfg
		logger.info('Loading detector...')
		self.predictor = VisualizationDemo(cfg)
		logger.info('Detector loaded successfully!')

	def setup_cfg(self, args):
		cfg = get_cfg()
		cfg.merge_from_file(args.config_file)
		# Set score_threshold for builtin models
		cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
		cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
		cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
		cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
		cfg.MODEL.DEVICE = args.device
		cfg.freeze()
		return cfg

	def postprocess_bezier(self, instances):
		beziers = np.abs(instances.beziers.numpy().astype("int"))
		scores = instances.scores.tolist()
		#texts = instances.texts
		texts = instances.recs

		result_boxes = []
		result_texts = []

		for bezier, score, text in zip(beziers, scores, texts):
			if score < self.args.rec_confidence_threshold: continue

			x1,y1 = bezier[:2]
			x2,y2 = bezier[6:8]
			x3,y3 = bezier[8:10]
			x4,y4 = bezier[-2:]

			result_boxes.append(list(map(int, [x1,y1,x2,y2,x3,y3,x4,y4])))
			result_texts.append(text)

		return result_boxes, result_texts

	def spotter(self, image):
		predictions, _ = self.predictor.run_on_image(image)
		instances = predictions['instances'].to(torch.device('cpu'))
		return self.postprocess_bezier(instances)

	def detect(self, image):
		result_boxes, _ = self.spotter(image)
		return result_boxes