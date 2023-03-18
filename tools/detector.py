from ultralytics import YOLO


class Detector:
    def __init__(self, player_prob_thresh, ball_prob_thresh, class_ids=[0, 32]):
        self.class_ids = class_ids
        self.player_prob_thresh = player_prob_thresh
        self.ball_prob_tresh = ball_prob_thresh
        self.model = YOLO('yolov8l.pt')

    def predict(self, image):

        outputs = self.model(image)
        return self.prepare_output(outputs)

    def prepare_output(self, outputs):
        classes = []
        boxes = []
        scores = []
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

        return {'classes': classes, 'boxes': boxes, 'scores': scores}
