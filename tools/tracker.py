import cv2
import numpy as np

from deep_sort_pytorch.deep_sort.sort import nn_matching, preprocessing
from deep_sort_pytorch.deep_sort.sort.detection import Detection
from deep_sort_pytorch.deep_sort.sort.tracker import Tracker as BaseTracker

from .encoder import ImageEncoder
from tools.reidentificator import Reidentificator

import os
import pickle
import math

class Tracker:
    def __init__(self,
                 cfg,
                 reID_model_path,
                 homography,
                 max_cosine_distance = 0.4, 
                 nn_budget = None,
                 nms_max_overlap = 0.7
                 ):
        
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.encoder = self.create_box_encoder(reID_model_path, batch_size=1)
        self.tracker = BaseTracker(metric)
        self.nms_max_overlap = nms_max_overlap
        self.tracking_output = []
        self.frame_num = 0
        self.reidentificator = Reidentificator(cfg)
        self.track_ids = []
        self.reid_tresh = cfg['REIDENTIFICATION_TRESH']
        self.reid_batch_size = cfg['REIDENTIFICATION_BATCH_SIZE']
        self.track_bundles = {}
        self.first_frame = True
        self.homography = homography
        current_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(os.path.abspath(current_path))
        background_path = os.path.join(parent_path, 'data', 'gui', 'maket.jpeg')
        background = cv2.imread(background_path)
        self.m_height, self.m_width, _ = background.shape

    def get_tracks(self):
        return self.tracker.tracks

    def predict(self, image, detections, db_conn):
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
        self.update_tracker_output(image, self.reid_tresh, db_conn)

    def check_point(self, x, y):
        center = np.array([x, y])
        point = np.array(center).reshape(-1,1,2).astype(np.float32)
        point = cv2.perspectiveTransform(point, self.homography)[0][0]
        return (0 <= point[0] <= self.m_width) and (0 <= point[1] <= self.m_height)

    def update_tracker_output(self, frame, tresh, db_conn):
        self.frame_num += 1
        tracks = []
        for track in self.tracker.tracks:
            # Get the track ID
            track_id = track.track_id

            box = track.to_tlbr().astype(int)
            xmin, ymin, xmax, ymax = box

            if self.check_point(xmin + (xmin - xmax) // 2, ymax):
                
                image = self.crop_image(xmin, ymin, xmax, ymax, frame)

                if not image.size == 0:
                    image_features = self.reidentificator.get_image_features(image)

                    # If track from first frame - bug fix
                    if self.first_frame:
                        self.track_bundles[track_id] = []

                    # If the track ID is already in the list of track IDs
                    if track_id in self.track_ids:
                        tracks.append({'id': track_id, 'box': box})
                    else:
                        prev_track_id = self.find_parent_track(track_id)

                        if prev_track_id:
                            tracks.append({'id': prev_track_id, 'box': box})
                            track_id = prev_track_id
                        else:
                            prev_track_id = self.find_previous_track(tresh, db_conn, image_features)
                            # If a previous track ID is found
                            if prev_track_id:
                                # Just update bundle
                                self.track_bundles[prev_track_id].append(track_id)
                                tracks.append({'id': prev_track_id, 'box': box})
                                track_id = prev_track_id
                            else:
                                tracks.append({'id': track_id, 'box': box})
                                self.track_ids.append(track_id)

                    self.insert_feature(track_id, image_features, db_conn)

        self.tracking_output.append({'frame_num': self.frame_num, 'tracks': tracks})
        self.first_frame = False
    
    def find_parent_track(self, track_id):
        for key, values in self.track_bundles.items():
            if track_id in values:
                return key
        return None
    
    def get_id_from_bundle(self, track_id):
        for parent_track_id, child_track_ids in self.track_bundles.items():
            if track_id == parent_track_id or track_id in child_track_ids:
                return parent_track_id
        
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

    def find_previous_track(self, tresh, db_conn, image_features):
        db_person_ids = self.get_distinct_person_ids(db_conn)
        tmp_features_dict = {}
        
        for person_id in db_person_ids:
            features_count = self.get_person_features_count(person_id, db_conn)
            num_iterations = math.ceil(features_count / self.reid_batch_size)
            for i in range(num_iterations):
                offset = i * self.reid_batch_size
                gallery_features = self.get_person_features(person_id, db_conn, self.reid_batch_size, offset)
                features = self.reidentificator.get_features_v2(image_features, gallery_features)
                min_feature = features[0].min()
                if min_feature <= tresh:
                    if person_id in tmp_features_dict:
                        if min_feature < tmp_features_dict[person_id]:
                            tmp_features_dict[person_id] = min_feature
                    else:
                        tmp_features_dict[person_id] = min_feature

        if tmp_features_dict:
            return min(tmp_features_dict, key=tmp_features_dict.get)
            
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
    
    def get_person_features_count(self, person_id, db_conn):
        cursor = db_conn.execute('SELECT COUNT(*) FROM features WHERE person_id = ?', (person_id,))
        return cursor.fetchone()[0]
    
    def get_person_features(self, person_id, db_conn, batch_size, offset):
        # select batch_size rows from the features table that have a matching person_id
        cursor = db_conn.execute('SELECT feature FROM features WHERE person_id = ? LIMIT ? OFFSET ?', (person_id, batch_size, offset))

        # retrieve the feature vectors from the rows and convert them back to numpy arrays using pickle
        features = []
        for row in cursor:
            feature_bytes = row[0]
            feature = pickle.loads(feature_bytes)
            features.append(feature)
        features = np.stack(features)
        features = np.squeeze(features, axis=1)  # remove extra dimension

        return features
    
    def insert_feature(self, person_id, feature, db_conn):
        # convert feature vector to bytes using pickle
        feature_bytes = pickle.dumps(feature)

        # insert the feature vector into the database
        db_conn.execute('INSERT INTO features (person_id, feature) VALUES (?, ?)', (person_id, feature_bytes))

        # commit the transaction and close the connection
        db_conn.commit()
    
    def get_distinct_person_ids(self, db_conn):
        cursor = db_conn.execute('SELECT DISTINCT person_id FROM features')
        # fetch all the results as a list of tuples
        results = cursor.fetchall()
        # extract the person_id values from the list of tuples
        person_ids = [result[0] for result in results]

        return person_ids