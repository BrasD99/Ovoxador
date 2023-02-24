from tools.pose import PoseEstimator
from tools.helpers import get_config, Body, Player, Frame, Dump, CustomEncoder
import cv2
import numpy as np
import json


cfg = get_config()
estimator = PoseEstimator(cfg)
image = cv2.imread('/Users/brasd99/Desktop/Dissertation/outputs/output2.jpg')

pose_output = estimator.process(image)

body = Body(pose_output)
player = Player(1, body, [])
frame = Frame(0, [player])
frames = [frame]
dump = Dump(frames)

with open('/Users/brasd99/Desktop/Dissertation/outputs/project/output.json', 'w') as f:
    json.dump(dump, f, cls=CustomEncoder)

print('dump created')