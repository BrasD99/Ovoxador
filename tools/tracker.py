import cv2
import numpy as np

from deep_sort_pytorch.deep_sort.sort import nn_matching, preprocessing
from deep_sort_pytorch.deep_sort.sort.detection import Detection
from deep_sort_pytorch.deep_sort.sort.tracker import Tracker as BaseTracker

from .encoder import ImageEncoder
from tools.reidentificator import Reidentificator

import os


class Tracker:
    def __init__(self,
                 cfg,
                 reID_model_path,
                 homography,
                 max_cosine_distance = 0.4, 
                 nn_budget = None,
                 nms_max_overlap = 0.7,
                 reid_tresh = 150
                 ):
        
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.encoder = self.create_box_encoder(reID_model_path, batch_size=1)
        self.tracker = BaseTracker(metric)
        self.nms_max_overlap = nms_max_overlap
        self.tracking_output = []
        self.frame_num = 0
        self.reidentificator = Reidentificator(cfg)
        self.images = {}
        self.track_ids = []
        self.reid_tresh = reid_tresh
        self.track_bundles = {}
        self.first_frame = True
        self.homography = homography
        self.inv_homography = np.linalg.inv(homography)
        current_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(os.path.abspath(current_path))
        background_path = os.path.join(parent_path, 'data', 'gui', 'maket.jpeg')
        background = cv2.imread(background_path)
        self.m_height, self.m_width, _ = background.shape

    def get_tracks(self):
        return self.tracker.tracks

    def predict(self, image, detections):
        classes = detections['classes']
        boxes = detections['boxes']
        scores = detections['scores']

        features = self.encoder(image, boxes)
        detections = [Detection(box, score, feature, class_id) for box, score, class_id, feature in zip(boxes, scores, classes, features)]
            
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.oid for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        self.tracker.predict()
        self.tracker.update(detections)
        self.update_tracker_output(image, self.reid_tresh)

    def check_point(self, x, y):
        center = np.array([x, y])
        point = np.array(center).reshape(-1,1,2).astype(np.float32)
        point = cv2.perspectiveTransform(point, self.homography)[0][0]
        return (0 <= point[0] <= self.m_width) and (0 <= point[1] <= self.m_height)

    def update_tracker_output(self, frame, tresh):
        self.frame_num += 1
        tracks = []
        track_ids = [track.track_id for track in self.tracker.tracks]
        for track in self.tracker.tracks:
            # Get the track ID
            track_id = track.track_id

            box = track.to_tlbr().astype(int)
            xmin, ymin, xmax, ymax = box

            if self.check_point(xmin + (xmin - xmax) // 2, ymax):
                
                image = self.crop_image(xmin, ymin, xmax, ymax, frame)

                if not image.size == 0:
                    
                    # If the track ID is already in the list of track IDs
                    if track_id in self.track_ids or self.first_frame:
                        self.track_ids.append(track_id)
                        tracks.append({'id': track_id, 'box': box})
                    else:
                        prev_track_id = self.find_parent_track(track_id)

                        if prev_track_id:
                            tracks.append({'id': prev_track_id, 'box': box})
                        else:
                            prev_track_id = self.find_previous_track(image, tresh)
                            # If a previous track ID is found
                            if prev_track_id and not prev_track_id in track_ids:
                                if prev_track_id not in self.track_bundles:
                                    self.track_bundles[prev_track_id] = [track_id]
                                else:
                                    self.track_bundles[prev_track_id].append(track_id)
                                tracks.append({'id': prev_track_id, 'box': box})
                                track_id = prev_track_id
                            # If a previous track ID is not found
                            else:
                                self.track_ids.append(track_id)
                                tracks.append({'id': track_id, 'box': box})

                    # Add the image for the current track ID
                    if track_id not in self.images:
                        self.images[track_id] = [image]
                    elif len(self.images[track_id]) < 8:
                        self.images[track_id].append(image)

        self.tracking_output.append({'frame_num': self.frame_num, 'tracks': tracks})
        self.first_frame = False
    
    def find_parent_track(self, track_id):
        for key, values in self.track_bundles.items():
            if track_id in values:
                return key
        return None
    
    def crop_image(self, xmin, ymin, xmax, ymax, frame):
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

    def find_previous_track(self, image, tresh):

        tmp_dict = {}
        for key in self.images:
            gallery_imgs = self.images[key]
            query_imgs = [image]
            features = self.reidentificator.get_features(query_imgs, gallery_imgs)
            min_feature = features[0].min()
            if min_feature <= tresh:
                tmp_dict[key] = min_feature

        if tmp_dict:
            return min(tmp_dict, key=tmp_dict.get)
            
        return None

    def extract_image_patch(self, image, bbox, patch_shape):
        bbox = np.array(bbox)
        if patch_shape is not None:
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width

        bbox[2:] += bbox[:2]
        bbox = bbox.astype(int)

        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image
    
    def create_box_encoder(self, model_filename, input_name="images",
                       output_name="features", batch_size=32):
        image_encoder = ImageEncoder(model_filename, input_name, output_name)
        image_shape = image_encoder.image_shape

        def encoder(image, boxes):
            image_patches = []
            for box in boxes:
                patch = self.extract_image_patch(image, box, image_shape[:2])
                if patch is None:
                    print("WARNING: Failed to extract image patch: %s." % str(box))
                    patch = np.random.uniform(
                        0., 255., image_shape).astype(np.uint8)
                image_patches.append(patch)
            image_patches = np.asarray(image_patches)
            return image_encoder(image_patches, batch_size)

        return encoder
    