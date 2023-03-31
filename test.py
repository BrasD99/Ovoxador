from tools.extractor import Extractor
from tools.helpers import load_homography
from tools.camera import Camera
from tools.helpers import get_config, create_output_directory, get_cameras_config, CustomEncoder, split_dict
from tools.texture import TextureExporter
from tools.pose import PoseEstimator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
from scipy.optimize import linear_sum_assignment
import numpy as np

f = open('/Users/brasd99/Desktop/Dissertation/outputs/project/dump/cameras.json')
cameras_players = json.load(f)
f.close()

# Define a function to compute the distance between player positions
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Define a function to perform data association between two cameras
def associate_players(cam1, cam2):
    # Create a cost matrix of distances between player positions
    cost_matrix = np.zeros((len(cam1['h_centers']), len(cam2['h_centers'])))
    for i in range(len(cam1['h_centers'])):
        for j in range(len(cam2['h_centers'])):
            cost_matrix[i, j] = distance(cam1['h_centers'][i][0], cam1['h_centers'][i][1],
                                          cam2['h_centers'][j][0], cam2['h_centers'][j][1])

    # Use the Hungarian algorithm to find the best match between players
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Create a list of matched player IDs
    matched_ids = []
    for i, j in zip(row_indices, col_indices):
        # Only consider matches with a low enough cost (distance)
        if cost_matrix[i, j] < MAX_DISTANCE:
            # Add the matched player IDs to the list
            matched_ids.append((cam1['id'], cam1['frame_ids'][i], cam2['id'], cam2['frame_ids'][j]))

    return matched_ids


# Define a maximum distance beyond which two player positions are considered unmatched
MAX_DISTANCE = 400

# Loop through all pairs of cameras
for i in range(len(cameras_players)):
    for j in range(i+1, len(cameras_players)):
        # Get the players from the two cameras
        cam1 = cameras_players[i]
        cam2 = cameras_players[j]

        # Create a list of matched player IDs
        matched_ids = []

        # Loop through all pairs of players from cam1 and cam2
        for k in range(len(cam1)):
            for l in range(len(cam2)):
                # Perform data association between the pairs of players
                matches = associate_players(cam1[k], cam2[l])

                # Add the matched player IDs to the list
                matched_ids.extend(matches)

        # Print the matched player IDs for the two cameras
        print(f"Matches between cameras {cam1['id']} and {cam2['id']}: {matched_ids}")




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
