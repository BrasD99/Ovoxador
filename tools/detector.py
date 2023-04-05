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
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

    def predict(self, image):
        classes = []
        boxes = []
        scores = []

        outputs = self.model(image)

        if self.use_latest:
            for box in outputs[0].boxes:
                class_id = int(box.cls)
                score = float(box.conf)
                if class_id in self.class_ids:
                    threshold = self.player_prob_tresh if class_id == 0 else self.ball_prob_tresh
                    if score >= threshold:
                        x1, y1, w, h = [float(coord) for coord in box.xywh[0]]
                        b_box = [x1 - w / 2, y1 - h / 2, w, h]
                        classes.append(class_id)
                        boxes.append(b_box)
                        scores.append(score)
        else:
            for i in range(outputs.n):
                class_ids = outputs.pred[i][:, -1].int()
                confidences = outputs.pred[i][:, 4].float()
                bboxes = outputs.pred[i][:, :4]

                for class_id, confidence, bbox in zip(class_ids, confidences, bboxes):
                    if class_id in self.class_ids:
                        threshold = self.player_prob_tresh if class_id == 0 else self.ball_prob_tresh
                        if confidence >= threshold:
                            x1, y1, x2, y2 = bbox.cpu().numpy()
                            w, h = x2 - x1, y2 - y1
                            b_box = [x1, y1, w, h]
                            classes.append(int(class_id))
                            boxes.append(b_box)
                            scores.append(float(confidence))

        return {'classes': classes, 'boxes': boxes, 'scores': scores}