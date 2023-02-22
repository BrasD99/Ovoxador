from tools.pose import PoseEstimator
from tools.helpers import get_config
import cv2

cfg = get_config()
estimator = PoseEstimator(cfg)
image = cv2.imread('/Users/brasd99/Desktop/Dissertation/outputs/output2.jpg')

output = estimator.process(image)

import torch
import smplx

# Load the SMPL model
model = smplx.create('/Users/brasd99/Desktop/Dissertation/Ovoxador/data/body_models/smpl/SMPL_NEUTRAL.pkl', model_type='smpl', gender='neutral')

joints3d = output['joints3d']

print(joints3d)

#from PARE.pare.utils.vis_utils import show_3d_pose

#show_3d_pose(joints3d[0])