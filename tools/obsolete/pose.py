import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class PoseEstimator:

    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.4)

    def process(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.pose.process(image)
