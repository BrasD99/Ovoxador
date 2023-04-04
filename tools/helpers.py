import json
import os
import cv2
import numpy as np
import shutil


def filter_detections(detections, filter_class_id):
    filtered_classes = []
    filtered_boxes = []
    filtered_scores = []
    classes = detections["classes"]
    boxes = detections["boxes"]
    scores = detections["scores"]

    for i, class_id in enumerate(classes):
        if class_id == filter_class_id:
            filtered_classes.append(class_id)
            filtered_boxes.append(boxes[i])
            filtered_scores.append(scores[i])
    return {"classes": filtered_classes, "boxes": filtered_boxes, "scores": filtered_scores}


def box_with_max_score(detections):
    classes = detections["classes"]
    if classes:
        boxes = detections["boxes"]
        scores = detections["scores"]

        max_box = None
        max_score = -float('inf')

        for c, b, s in zip(classes, boxes, scores):
            if s > max_score:
                max_score = s
                max_box = b

        max_box = np.array(max_box).astype(int)
        xmin, ymin, xmax, ymax = max_box[0], max_box[1], max_box[0] + \
            max_box[2], max_box[1] + max_box[3]
        return xmin, ymin, xmax, ymax
    return None


def save_frame(frame, filename):
    tmp_frame = frame.copy()
    tmp_frame = cv2.cvtColor(tmp_frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename, tmp_frame)


def get_config():
    current_path = os.getcwd()
    config_file_path = os.path.join(current_path, 'config.json')
    with open(config_file_path) as json_file:
        return json.load(json_file)


def get_cameras_config(cfg, camera_ids):
    default_params = cfg['DEFAULT_CAMERA_PARAMS']
    output = {}
    for camera_id in camera_ids:
        camera_overrides = cfg.get(
            'CAMERA_OVERRIDEN_PARAMS', {}).get(str(camera_id), {})
        camera_params = {**default_params, **camera_overrides}
        camera_params["MAX_VIDEO_LENGTH"] = cfg["MAX_VIDEO_LENGTH"]
        camera_params["REID_MODEL_PATH"] = cfg["REID_MODEL_PATH"]
        camera_params["DENSEPOSE_CONFIG"] = cfg["DENSEPOSE_CONFIG"]
        camera_params["DENSEPOSE_WEIGHTS_URL"] = cfg["DENSEPOSE_WEIGHTS_URL"]
        camera_params["LAMA_MODEL_PATH"] = cfg["LAMA_MODEL_PATH"]
        camera_params["TORCHREID_MODEL_PATH"] = cfg["TORCHREID_MODEL_PATH"]
        camera_params["TEXTURES_MODE"] = cfg["TEXTURES_MODE"]
        camera_params["TEXTURES_FREQ"] = cfg["TEXTURES_FREQ"]
        camera_params["OUTPUT_FRAMES_SPLIT"] = cfg["OUTPUT_FRAMES_SPLIT"]
        camera_params["USE_LATEST_DETECTOR"] = cfg["USE_LATEST_DETECTOR"]

        output[camera_id] = camera_params
    return output


def parse_iuv(result):
    i = result['pred_densepose'][0].labels.cpu().numpy().astype(float)
    uv = (result['pred_densepose'][0].uv.cpu().numpy() * 255.0).astype(float)
    iuv = np.stack((uv[1, :, :], uv[0, :, :], i))
    iuv = np.transpose(iuv, (1, 2, 0))
    return iuv


def parse_bbox(result):
    return result["pred_boxes_XYXY"][0].cpu().numpy()


def concat_textures(array):
    texture = []
    for i in range(4):
        tmp = array[6 * i]
        for j in range(6 * i + 1, 6 * i + 6):
            tmp = np.concatenate((tmp, array[j]), axis=1)
        texture = tmp if len(texture) == 0 else np.concatenate(
            (texture, tmp), axis=0)
    return texture


def interpolate_tex(tex):
    # code is adopted from https://github.com/facebookresearch/DensePose/issues/68
    valid_mask = np.array((tex.sum(0) != 0) * 1, dtype='uint8')
    radius_increase = 10
    kernel = np.ones((radius_increase, radius_increase), np.uint8)
    dilated_mask = cv2.dilate(valid_mask, kernel, iterations=1)
    region_to_fill = dilated_mask - valid_mask
    invalid_region = 1 - valid_mask
    actual_part_max = tex.max()
    actual_part_min = tex.min()
    actual_part_uint = np.array(
        (tex - actual_part_min) / (actual_part_max - actual_part_min) * 255, dtype='uint8')
    actual_part_uint = cv2.inpaint(actual_part_uint.transpose((1, 2, 0)), invalid_region, 1,
                                   cv2.INPAINT_TELEA).transpose((2, 0, 1))
    actual_part = (actual_part_uint / 255.0) * \
        (actual_part_max - actual_part_min) + actual_part_min
    # only use dilated part
    actual_part = actual_part * dilated_mask

    return actual_part


def get_texture(im, iuv, bbox, tex_part_size=200):
    # this part of code creates iuv image which corresponds
    # to the size of original image (iuv from densepose is placed
    # within pose bounding box).
    im = im.transpose(2, 1, 0) / 255
    image_w, image_h = im.shape[1], im.shape[2]
    bbox[2] = bbox[2] - bbox[0]
    bbox[3] = bbox[3] - bbox[1]
    x, y, w, h = [int(v) for v in bbox]
    bg = np.zeros((image_h, image_w, 3))
    bg[y:y + h, x:x + w, :] = iuv
    iuv = bg
    iuv = iuv.transpose((2, 1, 0))
    i, u, v = iuv[2], iuv[1], iuv[0]

    # following part of code iterate over parts and creates textures
    # of size `tex_part_size x tex_part_size`
    n_parts = 24
    texture = np.zeros((n_parts, 3, tex_part_size, tex_part_size))

    for part_id in range(1, n_parts + 1):
        generated = np.zeros((3, tex_part_size, tex_part_size))

        x, y = u[i == part_id], v[i == part_id]
        # transform uv coodrinates to current UV texture coordinates:
        tex_u_coo = (x * (tex_part_size - 1) / 255).astype(int)
        tex_v_coo = (y * (tex_part_size - 1) / 255).astype(int)

        # clipping due to issues encountered in denspose output;
        # for unknown reason, some `uv` coos are out of bound [0, 1]
        tex_u_coo = np.clip(tex_u_coo, 0, tex_part_size - 1)
        tex_v_coo = np.clip(tex_v_coo, 0, tex_part_size - 1)

        # write corresponding pixels from original image to UV texture
        # iterate in range(3) due to 3 chanels
        for channel in range(3):
            generated[channel][tex_v_coo,
                               tex_u_coo] = im[channel][i == part_id]

        # this part is not crucial, but gives you better results
        # (texture comes out more smooth)
        if np.sum(generated) > 0:
            generated = interpolate_tex(generated)

        # assign part to final texture carrier
        texture[part_id - 1] = generated[:, ::-1, :]

    # concatenate textures and create 2D plane (UV)
    tex_concat = np.zeros((24, tex_part_size, tex_part_size, 3))
    for i in range(texture.shape[0]):
        tex_concat[i] = texture[i].transpose(2, 1, 0)
    tex = concat_textures(tex_concat)

    return tex


def create_iuv(results, image):
    iuv = parse_iuv(results)
    bbox = parse_bbox(results)
    # image = cv2.imread(image)[:, :, ::-1]
    # image = image[:, :, ::-1]
    uv_texture = get_texture(image, iuv, bbox)
    # plot texture or do whatever you like
    uv_texture = uv_texture.transpose([1, 0, 2])
    return uv_texture
    # imageio.imwrite(outputName, uv_texture)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_warning(text):
    print(f"{bcolors.WARNING}WARNING: {text}{bcolors.ENDC}")


def print_error(text):
    print(f"{bcolors.FAIL}ERROR: {text}{bcolors.ENDC}")


def align_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.array([kp1[m.queryIdx].pt for m in matches], np.float32)
    dst_pts = np.array([kp2[m.trainIdx].pt for m in matches], np.float32)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))


def save_homography(homopraphy, filename):
    cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE).write(
        "homography", homopraphy)


def load_homography(filename):
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    return fs.getNode("homography").mat()


class Player:
    def __init__(self, id, homography):
        self.id = id
        self.homography = homography
        self.frame_ids = []
        self.bboxes = []
        self.centers = []
        self.h_centers = []

    def add_bbox(self, frame_id, bbox):
        self.frame_ids.append(frame_id)
        self.bboxes.append(bbox)
        xmin, _, xmax, ymax = bbox
        center = np.array([xmin + (xmax - xmin) // 2, ymax])
        self.centers.append(center)
        point = np.array(center).reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(
            point, self.homography)[0][0]
        self.h_centers.append(transformed_point)

    def is_frame_exists(self, frame_id):
        return frame_id in self.frame_ids

    def get_by_frame(self, frame_id):
        i = self.frame_ids.index(frame_id)
        return {'id': self.id, 'bbox': self.bboxes[i], 'center': self.centers[i], 'h_center': self.h_centers[i]}


class Dump:
    def __init__(self, frames):
        self.frames = frames


class Frame:
    def __init__(self, id, players):
        self.id = id
        self.players = players


class PlayerDump:
    def __init__(self, id, body, position):
        self.id = id
        self.body = body
        self.position = position


class Body:
    def __init__(self, pose_output):
        self.pose = pose_output['pose'][0]
        self.betas = pose_output['betas'][0]


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, PlayerDump):
            return obj.__dict__
        if isinstance(obj, Frame):
            return obj.__dict__
        if isinstance(obj, Body):
            return obj.__dict__
        if isinstance(obj, Dump):
            return obj.__dict__
        if isinstance(obj, Player):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)


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
            missing_frame_ids = set(
                camera_player.frame_ids) - {frame_id for frame_id in main_player.frame_ids}
            # Getting remaining frames
            remaining_frame_ids = set(
                [frame_id for frame_id in camera_player.frame_ids if frame_id not in missing_frame_ids])

            if len(remaining_frame_ids) >= 0:
                m_centers = []
                c_centers = []
                for remaining_frame_id in remaining_frame_ids:
                    m_center = main_player.get_by_frame(
                        remaining_frame_id)['h_center']
                    c_center = camera_player.get_by_frame(
                        remaining_frame_id)['h_center']
                    m_centers.append(m_center)
                    c_centers.append(c_center)

                similarity = average_distance(
                    np.array(m_centers), np.array(c_centers))
                if similarity <= similarity_tresh:
                    output[main_player.id][camera_player.id] = {'frames': {
                        f_id + 1 for f_id in remaining_frame_ids}, 'similarity': similarity}

    for main_player_id in output:
        output[main_player_id] = dict(
            sorted(output[main_player_id].items(), key=lambda item: item[1]['similarity']))
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
        person_two_ids = [k[1]
                          for k in connections.keys() if k[0] == person_one_id]
        for person_two_id in person_two_ids:
            if not person_two_id in blocked_person_ids:
                result[person_one_id] = {
                    'id': person_two_id, 'similarity': connections[person_one_id, person_two_id]['similarity']}
                blocked_person_ids.append(person_two_id)
                break

    return result


def map_to_main_camera(aligned_arr):
    output = {}
    for camera_id in range(len(aligned_arr)):
        for main_camera_player_id, connections in aligned_arr[camera_id].items():
            for camera_player_id in connections.keys():
                if not main_camera_player_id in output:
                    output[main_camera_player_id] = []

                output[main_camera_player_id].append({
                    'camera_id': camera_id + 1,
                    'id': camera_player_id,
                    'frames': connections[camera_player_id]['frames'],
                    'similarity': connections[camera_player_id]['similarity']
                })
    for key, value in output.items():
        sorted_value = sorted(value, key=lambda x: x['similarity'])
        output[key] = sorted_value

    return output


def get_blocked_frames(connections):
    output = []
    for connection in connections:
        output = output + list(connection['frames'])
    return output


def get_optimal_main_connections(main_connections):
    blocked = []
    output = {}

    for player_id in main_connections.keys():
        if len(main_connections[player_id]) > 0:
            for connection in main_connections[player_id]:
                if not player_id in output:
                    output[player_id] = []
                if not (connection['camera_id'], connection['id']) in blocked:
                    blocked.append((connection['camera_id'], connection['id']))
                    blocked_frames = get_blocked_frames(output[player_id])
                    available_frames = [
                        num for num in connection['frames'] if num not in blocked_frames]
                    if available_frames:
                        output[player_id].append({
                            'camera_id': connection['camera_id'],
                            'id': connection['id'],
                            'frames': available_frames
                        })
    return dict(sorted(output.items()))


def get_players_for_frame(pose_output_dict, frame_id):
    output = {}
    for player_id, inner_data in pose_output_dict.items():
        for inner_player_data in inner_data:
            if inner_player_data['frame'] == frame_id:
                output[player_id] = inner_player_data
    return output


def map_to_frames_output(pose_output_dict, ball_positions_dict, frames_count):
    output_frames = {}

    for frame_id in range(frames_count):
        players_data = get_players_for_frame(pose_output_dict, frame_id + 1)
        ball_position = ball_positions_dict.get(frame_id + 1)
        output_frames[frame_id] = {
            'ball_position': ball_position,
            'players': [{
                'id': player_id,
                'camera_id': player_data['camera_id'],
                'pose': player_data['pose'],
                'position': player_data['player_position']
            } for player_id, player_data in players_data.items()]
        }

    return output_frames


def map_to_general_output(frames_output, cameras_positions):
    return {
        'cameras': cameras_positions,
        'frames': frames_output
    }


def get_cameras_players(cameras, homographies):
    cameras_players = []

    for camera_id, camera in enumerate(cameras):
        camera_players = []
        player_tracks = camera.get_player_tracks()
        for frame_num, frame_tracks in enumerate(player_tracks):
            for track in frame_tracks['tracks']:
                track_id = track['id']
                box = track['box'].astype(int)
                player = next(
                    (x for x in camera_players if x.id == track_id), None)
                if not player:
                    player = Player(track_id, homographies[camera_id])
                    camera_players.append(player)

                player.add_bbox(frame_num, box)

        cameras_players.append(camera_players)

    return cameras_players


def get_aligned_players(cameras_players, tresh=200):
    aligned_arr = []
    for i in range(1, len(cameras_players)):
        aligned_players = align_players(
            cameras_players[0], cameras_players[i], tresh)
        aligned_arr.append(aligned_players)
    return aligned_arr


def get_optimal_players_connections(aligned_arr):
    main_connections = map_to_main_camera(aligned_arr)
    optimal_connections = get_optimal_main_connections(main_connections)
    return optimal_connections


def get_output_dict(pose_dict, ball_positions, cameras_positions, main_camera):
    frames_count = len(main_camera.frames)
    output = map_to_frames_output(pose_dict, ball_positions, frames_count)
    output = map_to_general_output(output, cameras_positions)
    return output


def get_players_images_v2(camera, player_id):
    other_camera_player_tracks = camera.get_player_tracks()
    return get_player_images_by_id(camera, other_camera_player_tracks, player_id)


def get_players_images_on_frames(camera, player_id, frame_ids):
    other_camera_player_tracks = camera.get_player_tracks()
    other_camera_player_tracks = [
        d for d in other_camera_player_tracks if d['frame_num'] in frame_ids]
    return get_player_images_by_id(camera, other_camera_player_tracks, player_id)


def create_output_directory(output_path, videos):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    videos_path = os.path.join(output_path, 'videos')
    frames_path = os.path.join(output_path, 'frames')
    textures_path = os.path.join(output_path, 'textures')
    homographies_path = os.path.join(output_path, 'homography')
    analytics_path = os.path.join(output_path, 'analytics')
    heatmaps_path = os.path.join(output_path, 'heatmaps')
    dump_folder = os.path.join(output_path, 'dump')
    path_arr = [videos_path, frames_path, textures_path,
                homographies_path, analytics_path, heatmaps_path, 
                dump_folder]
    create_directories(path_arr, videos, frames_path, videos_path)
    return {
        'videos_path': videos_path,
        'frames_path': frames_path,
        'textures_path': textures_path,
        'homographies_path': homographies_path,
        'analytics_path': analytics_path,
        'heatmaps_path': heatmaps_path,
        'dump_folder': dump_folder
    }


def create_directories(path_arr, videos, frames_path, videos_path):
    for i in range(len(videos)):
        frame_path = os.path.join(frames_path, f'camera_{i}')
        path_arr.append(frame_path)

    for path in path_arr:
        if not os.path.exists(path):
            os.makedirs(path)

    for i, video_path in enumerate(videos):
        _, ext = os.path.splitext(video_path)
        shutil.copy(video_path, f'{videos_path}/camera_{i}{ext}')


def get_players_images(cameras, connections):
    main_camera = cameras[0]
    main_camera_player_tracks = main_camera.get_player_tracks()
    players_images = {}
    for i in range(len(connections)):
        other_camera = cameras[i + 1]
        other_camera_player_tracks = other_camera.get_player_tracks()
        for main_camera_player_id in connections[i]:
            other_camera_player_id = connections[i][main_camera_player_id]['id']
            main_player_images = get_player_images_by_id(
                main_camera, main_camera_player_tracks, main_camera_player_id)
            other_player_images = get_player_images_by_id(
                other_camera, other_camera_player_tracks, other_camera_player_id)
            if not main_camera_player_id in players_images:
                players_images[main_camera_player_id] = []
            players_images[main_camera_player_id].extend(main_player_images)
            players_images[main_camera_player_id].extend(other_player_images)

    return players_images


def get_player_images_by_id(camera, player_tracks, track_id):
    output = []
    for frame_num, frame_tracks in enumerate(player_tracks):
        src_frame = camera.frames[frame_num]
        for track in frame_tracks['tracks']:
            if track['id'] == track_id:
                box = track['box'].astype(int)
                xmin, ymin, xmax, ymax = box
                image = crop_image(xmin, ymin, xmax, ymax, src_frame)
                # image = cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB)
                output.append(image)
    return output


def crop_image(xmin, ymin, xmax, ymax, frame):
    image = frame.copy()
    # Check if ymin is less than zero
    if ymin < 0:
        ymin = 0

    # Check if xmin is less than zero
    if xmin < 0:
        xmin = 0

    # Check if ymax is greater than the number of rows in the image
    if ymax > image.shape[0]:
        ymax = image.shape[0]

    # Check if xmax is greater than the number of columns in the image
    if xmax > image.shape[1]:
        xmax = image.shape[1]

    return image[ymin:ymax, xmin:xmax]


def get_ball_positions(cameras, homography_array):
    output = {}
    for camera_id, camera in enumerate(cameras):
        ball_detections = camera.get_ball_detections()
        for frame_num, ball_detection in enumerate(ball_detections):
            if not frame_num + 1 in output:
                output[frame_num + 1] = None
                if len(ball_detection['scores']) > 0:
                    max_index = ball_detection['scores'].index(
                        max(ball_detection['scores']))
                    bbox = ball_detection['boxes'][max_index]
                    xmin, xmax, ymax = bbox[0], bbox[0] + \
                        bbox[2], bbox[1] + bbox[3]
                    center = np.array([xmin + (xmax - xmin) // 2, ymax])
                    point = np.array(center).reshape(-1, 1,
                                                     2).astype(np.float32)
                    transformed_point = cv2.perspectiveTransform(
                        point, homography_array[camera_id])[0][0]
                    output[frame_num + 1] = transformed_point.tolist()
    return output

def split_dict(input, n):
    items = list(input.items())
    sublists = [items[i:i+n] for i in range(0, len(items), n)]
    output = []
    for sublist in sublists:
        output.append(dict(sublist))
    return output
