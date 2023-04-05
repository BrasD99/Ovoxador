from tools.detector import Detector
import cv2

detectors = [Detector(0.1, 0.1, use_latest=True), Detector(0.1, 0.1, use_latest=False)]
img = cv2.imread('/Users/brasd99/Desktop/Dissertation/outputs/output2.jpg')

for detector in detectors:
    output = detector.predict(img)
    img_copy = img.copy()
    for box in output['boxes']:
        x, y, w, h = map(int, box)
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Show the image in a cv2 window
    cv2.imshow('Bounding Box Image', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

