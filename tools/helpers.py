import json
import os

import cv2
import imageio
import numpy as np
import torch


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
    return { "classes": filtered_classes, "boxes": filtered_boxes, "scores": filtered_scores }

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
        xmin, ymin, xmax, ymax = max_box[0], max_box[1], max_box[0] + max_box[2], max_box[1] + max_box[3]
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
        camera_overrides = cfg.get('CAMERA_OVERRIDEN_PARAMS', {}).get(str(camera_id), {})
        camera_params = {**default_params, **camera_overrides}
        camera_params["MAX_VIDEO_LENGTH"] = cfg["MAX_VIDEO_LENGTH"]
        camera_params["REID_MODEL_PATH"] = cfg["REID_MODEL_PATH"]
        camera_params["DENSEPOSE_CONFIG"] = cfg["DENSEPOSE_CONFIG"]
        camera_params["DENSEPOSE_WEIGHTS_URL"] = cfg["DENSEPOSE_WEIGHTS_URL"]
        camera_params["LAMA_MODEL_PATH"] = cfg["LAMA_MODEL_PATH"]
        camera_params["TORCHREID_MODEL_PATH"] = cfg["TORCHREID_MODEL_PATH"]
        camera_params["TEXTURES_MODE"] = cfg["TEXTURES_MODE"]

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
        texture = tmp if len(texture) == 0 else np.concatenate((texture, tmp), axis=0)
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
    actual_part_uint = np.array((tex - actual_part_min) / (actual_part_max - actual_part_min) * 255, dtype='uint8')
    actual_part_uint = cv2.inpaint(actual_part_uint.transpose((1, 2, 0)), invalid_region, 1,
                               cv2.INPAINT_TELEA).transpose((2, 0, 1))
    actual_part = (actual_part_uint / 255.0) * (actual_part_max - actual_part_min) + actual_part_min
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
            generated[channel][tex_v_coo, tex_u_coo] = im[channel][i == part_id]
        
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
    #image = cv2.imread(image)[:, :, ::-1]
    #image = image[:, :, ::-1]
    uv_texture = get_texture(image, iuv, bbox)
    # plot texture or do whatever you like
    uv_texture = uv_texture.transpose([1,0,2])
    return uv_texture
    #imageio.imwrite(outputName, uv_texture)

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
    cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE).write("homography", homopraphy)

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
        point = np.array(center).reshape(-1,1,2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(point, self.homography)[0][0]
        self.h_centers.append(transformed_point)
    
    def is_frame_exists(self, frame_id):
        return frame_id in self.frame_ids
    
    def get_by_frame(self, frame_id):
        i = self.frame_ids.index(frame_id)
        return { 'id': self.id, 'bbox': self.bboxes[i], 'center': self.centers[i], 'h_center': self.h_centers[i] }

