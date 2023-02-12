from tools.helpers import load_homography, Player
from tools.camera import Camera
from tools.helpers import get_config, get_cameras_config, box_with_max_score, save_frame, save_homography, create_iuv
from tools.texture import TextureExporter
import cv2
import numpy as np
import glob
import random
import shutil
import concurrent.futures
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

if os.path.exists(output_path):
    shutil.rmtree(output_path)

videos_path = os.path.join(output_path, 'videos')
frames_path = os.path.join(output_path, 'frames')
textures_path = os.path.join(output_path, 'textures')
homographies_path = os.path.join(output_path, 'homography')
analytics_path = os.path.join(output_path, 'analytics')

path_arr = [videos_path, frames_path, textures_path, homographies_path, analytics_path]

for i in range(len(videos)):
    frame_path = os.path.join(frames_path, f'camera_{i}')
    path_arr.append(frame_path)

for path in path_arr:
    if not os.path.exists(path):
        os.makedirs(path)

for i, video_path in enumerate(videos):
    base, ext = os.path.splitext(video_path)
    shutil.copy(video_path, f'{videos_path}/camera_{i}{ext}')

main_camera = Camera(
    video_file=main_path, 
    camera_id=0,
    homography=homographies[0], 
    cfg=cameras_cfg[0])

tl_camera = Camera(
    video_file=tl_path, 
    camera_id=1, 
    homography=homographies[1], 
    cfg=cameras_cfg[1])

tr_camera = Camera(
    video_file=tr_path,
    camera_id=2,
    homography=homographies[2],
    cfg=cameras_cfg[2])

cameras = [main_camera, tl_camera, tr_camera]

extractor = Extractor(cfg=cfg)

for camera in cameras:
    camera.process()

extractor.export_bboxes(cameras, analytics_path)

cameras_players = []

for camera_id, camera in enumerate(cameras):
    camera_players = []
    player_tracks = camera.get_player_tracks()
    for frame_num, frame_tracks in enumerate(player_tracks):
        for track in frame_tracks['tracks']:
            track_id = track['id']
            box = track['box'].astype(int)
            player = next((x for x in camera_players if x.id == track_id), None)
            if not player:
                player = Player(track_id, homographies[camera_id])
                camera_players.append(player)
            
            player.add_bbox(frame_num, box)

    cameras_players.append(camera_players)

main_camera_players = cameras_players[0]

extractor.export_centers(cameras, homographies, cameras_players, analytics_path)

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def average_distance(array1, array2):
    distances = []
    for p1 in array1:
        for p2 in array2:
            distances.append(euclidean_distance(p1, p2))
    return np.mean(distances)

def align_players(main_camera_players, camera_players, similarity_tresh):
    output = {}
    for main_player in main_camera_players:
        output[main_player.id] = {}
        for camera_player in camera_players:
            # Getting missed frames
            missing_frame_ids = set(camera_player.frame_ids) - {frame_id for frame_id in main_player.frame_ids}
            # Getting remaining frames
            remaining_frame_ids = set([frame_id for frame_id in camera_player.frame_ids if frame_id not in missing_frame_ids])

            if len(remaining_frame_ids) >= 5:
                m_centers = []
                c_centers = []
                for remaining_frame_id in remaining_frame_ids:
                    m_center = main_player.get_by_frame(remaining_frame_id)['h_center']
                    c_center = camera_player.get_by_frame(remaining_frame_id)['h_center']
                    m_centers.append(m_center)
                    c_centers.append(c_center)
                
                similarity = average_distance(np.array(m_centers), np.array(c_centers))
                if similarity <= similarity_tresh:
                    output[main_player.id][camera_player.id] = { 'frames': {f_id + 1 for f_id in remaining_frame_ids}, 'similarity': similarity }

    for main_player_id in output:
        output[main_player_id] = dict(sorted(output[main_player_id].items(), key=lambda item: item[1]['similarity']))
    return output

def get_min_dist(aligned_players):
    result = {}
    for main_player_id, connections in aligned_players.items():
        for camera_player_id, mse in connections.items():
            if not (main_player_id, camera_player_id) in result:
                result[(main_player_id, camera_player_id)] = mse
            else:
                if mse < result[(main_player_id, camera_player_id)]:
                    result[(main_player_id, camera_player_id)] = mse
    return result

def get_optimal_connections(connections):
    result = {}
    blocked_person_ids = []
    person_one_ids = set([k[0] for k in connections.keys()])
    for person_one_id in person_one_ids:
        person_two_ids = [k[1] for k in connections.keys() if k[0] == person_one_id]
        for person_two_id in person_two_ids:
            if not person_two_id in blocked_person_ids:
                result[person_one_id] = {'id': person_two_id, 'similarity': connections[person_one_id, person_two_id]['similarity']}
                blocked_person_ids.append(person_two_id)
                break

    return result

connections = []

for i in range(1, len(cameras_players)):
    aligned_players = align_players(main_camera_players, cameras_players[i], 100)
    result = get_min_dist(aligned_players)
    result = get_optimal_connections(result)
    extractor.export_analytics(i, cameras, result, analytics_path)
    connections.append(result)

texture_exporter = TextureExporter(cfg)
# Saving players textures
extractor.export_players_textures(cameras, connections, texture_exporter, textures_path)

# Saving pitch texture
#extractor.export_pitch(cameras[0], homographies[0], textures_path)
# Saving frames
extractor.export_frames(cameras, frames_path)
# Saving homography matrixes
extractor.export_homography_array(homographies, homographies_path)