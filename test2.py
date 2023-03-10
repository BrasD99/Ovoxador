import pickle
from tools.helpers import load_homography
import numpy as np

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

            if len(remaining_frame_ids) >= 0:
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

# Load the list of objects from the file using pickle
with open('/Users/brasd99/Desktop/Dissertation/outputs/cameras_players.pickle', 'rb') as f:
    cameras_players = pickle.load(f)

main_h = load_homography('/Users/brasd99/Downloads/homography_0.yml')
tl_h = load_homography('/Users/brasd99/Downloads/homography_1.yml')
tr_h = load_homography('/Users/brasd99/Downloads/homography_2.yml')

homographies = [main_h, tl_h, tr_h]

def map_to_main_camera(aligned_arr):
    output = {}
    for camera_id in range(len(aligned_arr)):
        for camera_player_id, connections in aligned_arr[camera_id].items():
            for main_camera_player_id in connections.keys():
                if not main_camera_player_id in output:
                    output[main_camera_player_id] = []
                
                output[main_camera_player_id].append({
                    'camera_id': camera_id + 1,
                    'id': camera_player_id,
                    'frames': connections[main_camera_player_id]['frames'],
                    'similarity': connections[main_camera_player_id]['similarity']
                })
    for key, value in output.items():
        sorted_value = sorted(value, key=lambda x: (len(x['frames']), x['similarity']))
        output[key] = sorted_value

    return output

def get_optimal_main_connections(main_connections):
    blocked = []
    output = {}
    for player_id in main_connections.keys():
        if len(main_connections[player_id]) > 0:
            connection = main_connections[player_id][0]
            if not (connection['camera_id'], connection['id']) in blocked:
                blocked.append((connection['camera_id'], connection['id']))
                output[player_id] = {
                    'camera_id': connection['camera_id'],
                    'id': connection['id']
                }
    return dict(sorted(output.items()))


#1. Соотносим игроков с других камер с игроками с основной камеры, 
#запоминаем общие ид кадров и степень схожести по средней mse.

aligned_arr = []
for i in range(1, len(cameras_players)):
    aligned_players = align_players(cameras_players[0], cameras_players[i], 200)
    aligned_arr.append(aligned_players)

# Ищем оптимальные связки
main_connections = map_to_main_camera(aligned_arr)
optimal_connections = get_optimal_main_connections(main_connections)

print(optimal_connections)