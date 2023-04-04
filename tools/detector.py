from ultralytics import YOLO
import torch

class Detector:
    def __init__(self, player_prob_tresh, ball_prob_tresh, class_ids=[0, 32], use_latest=False):
        self.class_ids = class_ids
        self.player_prob_tresh = player_prob_tresh
        self.ball_prob_tresh = ball_prob_tresh
        self.use_latest = use_latest
        if self.use_latest:
            self.model = YOLO('yolov8l.pt')
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def predict(self, image):
        classes = []
        boxes = []
        scores = []

        outputs = self.model(image)

        if self.use_latest:
            for box in outputs[0].boxes:
                class_id = box.cls.item()
                score = box.conf.item()
                if class_id in self.class_ids:
                    if class_id == 0:
                        if score >= self.player_prob_thresh:
                            x1, y1, w, h = [box.xywh[0][i].item()
                                            for i in range(4)]
                            b_box = [x1 - w / 2, y1 - h / 2, w, h]
                            classes.append(class_id)
                            boxes.append(b_box)
                            scores.append(score)
                    else:
                        if score >= self.ball_prob_tresh:
                            x1, y1, w, h = [box.xywh[0][i].item()
                                            for i in range(4)]
                            b_box = [x1 - w / 2, y1 - h / 2, w, h]
                            classes.append(class_id)
                            boxes.append(b_box)
                            scores.append(score)
        else:
            for _, row in outputs.pandas().xywh[0].iterrows():
                if row['class'] in self.class_ids:
                    if row['class'] == 0:
                        if row['confidence'] >= self.player_prob_thresh:
                            b_box = [row['xcenter'], row['ycenter'], row['width'], row['height']]
                            classes.append(row['class'])
                            boxes.append(b_box)
                            scores.append(row['confidence'])
                    else:
                        if row['confidence'] >= self.ball_prob_tresh:
                            b_box = [row['xcenter'], row['ycenter'], row['width'], row['height']]
                            classes.append(row['class'])
                            boxes.append(b_box)
                            scores.append(row['confidence'])

        return {'classes': classes, 'boxes': boxes, 'scores': scores}