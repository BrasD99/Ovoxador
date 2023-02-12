import os
from typing import Any, Dict
import cv2
import numpy as np
from tqdm import tqdm
from .helpers import box_with_max_score

import torch
from densepose import add_densepose_config
from densepose.structures import (DensePoseChartPredictorOutput,
                                  DensePoseEmbeddingPredictorOutput)
from densepose.vis.extractor import (DensePoseOutputsExtractor,
                                     DensePoseResultExtractor)
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from .helpers import align_images
from .lama import LamaProcessor

class TextureExporter:
    def __init__(self, cfg):
        self.config = self.get_config(cfg['DENSEPOSE_CONFIG'], cfg['DENSEPOSE_WEIGHTS_URL'])
        self.predictor = DefaultPredictor(self.config)
        self.lama = LamaProcessor(cfg)

    def execute(self, image):
        context = { 'results': [] }
        with torch.no_grad():
            outputs = self.predictor(image)["instances"]
            self.execute_on_outputs(context, outputs)
        return context["results"]
    
    def get_config(self, config_fpath, model_fpath):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.MODEL.DEVICE = "cpu"
        cfg.freeze()
        return cfg
    
    def execute_on_outputs(
        self,
        context: Dict[str, Any], 
        outputs: Instances):
        result = {}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]
        context["results"].append(result)
    
    def hide_boxes(self, camera, frame, player_tracks, ball_detections, image, use_sift = False):
        first_frame = cv2.cvtColor(camera.frames[0], cv2.COLOR_BGR2RGB)
        src_frame = frame.copy()
        #src_frame = cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB)

        track_bboxes = [track['box'].astype(int) for track in player_tracks["tracks"]]
        detection_bboxes = box_with_max_score(ball_detections)

        if use_sift:
            src_frame = align_images(first_frame, src_frame)
        
        for xmin, ymin, xmax, ymax in track_bboxes:
            src_frame[ymin:ymax, xmin:xmax, :] = 0
        if detection_bboxes:
            xmin, ymin, xmax, ymax = detection_bboxes
            src_frame[ymin:ymax, xmin:xmax, :] = 0

        non_zero = (src_frame != 0) & (image == 0)
        image[non_zero] = src_frame[non_zero]

        return image
    
    def extract_main_color(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        main_color = np.argmax(hist)

        main_color = np.uint8([[[main_color, 255, 255]]])
        main_color = cv2.cvtColor(main_color, cv2.COLOR_HSV2BGR)

        return main_color
    
    def extract_contours(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours
    
    def draw_contours(self, image, contours, main_color):
        result = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        result[:] = main_color
        cv2.drawContours(result, contours, -1, (255, 255, 255), 2)
        return result
    
    def prepare_pitch_texture(self, background_image, original_image, transformed_image):
        main_color = self.extract_main_color(original_image)
        contours = self.extract_contours(background_image)
        return self.draw_contours(transformed_image, contours, main_color)
    
    def combine_textures(self, background, transformed_img, alpha = 1):
        mask = (transformed_img != 0).astype(np.float32) * alpha
        background = background.astype(np.float32) * (1 - mask) + transformed_img.astype(np.float32) * mask

        return background.astype(np.uint8)
    
    def extract_pitch(self, background_image, original_image, transformed_image):
        background = self.prepare_pitch_texture(background_image, original_image, transformed_image)
        return self.combine_textures(background, transformed_image)
    
    def get_pitch_texture(self, camera, homography, use_sift = False):
        current_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(os.path.abspath(current_path))
        background_path = os.path.join(parent_path, 'data', 'gui', 'maket.jpeg')
        background = cv2.imread(background_path)

        player_tracks = camera.get_player_tracks()
        ball_detections = camera.get_ball_detections()

        height, width, channels = camera.frames[0].shape
        m_height, m_width, _ = background.shape
        final = np.zeros((height, width, channels), dtype=camera.frames[0].dtype)

        for i in tqdm(range(len(camera.frames)), desc='Processing frames'):
            final = self.hide_boxes(camera, camera.frames[i], player_tracks[i], ball_detections[i], final, use_sift)

        transformed_img = cv2.warpPerspective(final, homography, (m_width, m_height))

        return self.extract_pitch(background, final, transformed_img)
    
    def get_pitch_texture_lama(self, frame, mask, homography):
        src_frame = frame.copy()
        current_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(os.path.abspath(current_path))
        background_path = os.path.join(parent_path, 'data', 'gui', 'maket.jpeg')
        background = cv2.imread(background_path)
        m_height, m_width, _ = background.shape
        output = self.lama.process(src_frame, mask)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        transformed_output = cv2.warpPerspective(output, homography, (m_width, m_height))
        return self.extract_pitch(background, output, transformed_output)