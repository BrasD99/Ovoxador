from tools.extractor import Extractor
from tools.helpers import load_homography
from tools.camera import Camera
from tools.helpers import get_config, create_output_directory, get_cameras_config
from tools.texture import TextureExporter
from tools.pose import PoseEstimator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#f = open('/Users/brasd99/Desktop/Dissertation/outputs/project/dump/cameras.json')
#cameras_players = json.load(f)
#f.close()

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

cameras_locations = [
    [36.47642517089844, 417.5539245605469],
    [1057.81640625, 40.539215087890625],
    [1050.5211181640625, 786.4607543945312]
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
    cameras_locations=cameras_locations,
    homographies=homographies,
    texture_exporter=texture_exporter,
    pose_estimator=pose_estimator,
    export_dict=export_dict,
    output_src_dict=output_src_dict)