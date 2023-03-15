from tools.helpers import load_homography
from tools.camera import Camera
from tools.helpers import get_config, create_output_directory, get_cameras_config
from tools.texture import TextureExporter
from tools.pose import PoseEstimator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tools.extractor import Extractor

cfg = get_config()

cameras_count = 3

cameras_cfg = get_cameras_config(cfg, [i for i in range(cameras_count)])

main_h = load_homography('/Users/brasd99/Downloads/homography_0.yml')
tl_h = load_homography('/Users/brasd99/Downloads/homography_1.yml')
tr_h = load_homography('/Users/brasd99/Downloads/homography_2.yml')

homographies = [main_h, tl_h, tr_h]

main_path = '/Users/brasd99/Desktop/Dissertation/Ovoxador/data/videos/camera_0.mov'
tl_path = '/Users/brasd99/Desktop/Dissertation/Ovoxador/data/videos/camera_1.mov'
tr_path = '/Users/brasd99/Desktop/Dissertation/Ovoxador/data/videos/camera_2.mov'

videos = [main_path, tl_path, tr_path]

output_path = '/Users/brasd99/Desktop/Dissertation/outputs/project'

output_src_dict = create_output_directory(output_path, videos)

cameras = [
    Camera(video_file=videos[i],
           camera_id=i,
           homography=homographies[i],
           cfg=cameras_cfg[i])
    for i in range(cameras_count)
]

extractor = Extractor(cfg=cfg)

for camera in cameras:
    camera.process()

export_dict = {
    'boxes': True,
    'centers': True,
    'dump': True,
    'frames': True,
    'homography': True,
    'pitch_texture': True,
    'players_texture': True
}

texture_exporter = TextureExporter(cfg)
pose_estimator = PoseEstimator(cfg)

extractor.export_all(
    cameras=cameras, 
    homographies=homographies,
    texture_exporter=texture_exporter,
    pose_estimator=pose_estimator,
    export_dict=export_dict,
    output_src_dict=output_src_dict)