import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

img = cv2.imread('/Users/brasd99/Downloads/Person.jpg')

def is_full_body(image, bbox):
    # Extract ROI from image using detector's bounding box
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]

    # Run MediaPipe Pose on ROI to estimate pose
    pose_results = pose_estimator.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    # Check if all major body parts are present and correctly estimated
    if pose_results.pose_landmarks is not None:
        landmarks = pose_results.pose_landmarks.landmark
        
        output = {
            'LEFT_SHOULDER': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility,
            'RIGHT_SHOULDER': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility,
            'LEFT_HIP': landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility,
            'RIGHT_HIP': landmarks[mp_pose.PoseLandmark.RIGHT_HIP].visibility,
            'LEFT_KNEE': landmarks[mp_pose.PoseLandmark.LEFT_KNEE].visibility,
            'RIGHT_KNEE': landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].visibility,
            'NOSE': landmarks[mp_pose.PoseLandmark.NOSE].visibility
        }

        print(output)

        if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.5 and \
           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.5 and \
           landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility > 0.5 and \
           landmarks[mp_pose.PoseLandmark.RIGHT_HIP].visibility > 0.5 and \
           landmarks[mp_pose.PoseLandmark.LEFT_KNEE].visibility > 0.5 and \
           landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].visibility > 0.5 and \
           landmarks[mp_pose.PoseLandmark.NOSE].visibility > 0.5:
            return True
        
    return False

bbox = [0, 0, 57, 73]
output = is_full_body(img, bbox)

print(output)