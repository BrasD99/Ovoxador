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

import mediapipe as mp


class Tracker:
    def __init__(self,
                 cfg,
                 reID_model_path,
                 homography,
                 max_cosine_distance=0.4,
                 nn_budget=None,
                 nms_max_overlap=0.7
                 ):

        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.encoder = self.create_box_encoder(reID_model_path, batch_size=1)
        self.tracker = BaseTracker(metric)
        self.nms_max_overlap = nms_max_overlap
        self.tracking_output = []
        self.frame_num = 0
        self.reidentificator = Reidentificator(cfg)
        self.mp_pose = mp.solutions.pose
        self.pose_estimator = self.mp_pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5)
        self.track_ids = []
        self.reid_tresh = cfg['REIDENTIFICATION_TRESH']
        self.reid_batch_size = cfg['REIDENTIFICATION_BATCH_SIZE']
        self.min_box_size = cfg['BOX_SIZE_TRESHTOLD']
        self.min_box_width = cfg['MIN_BOX_WIDTH']
        self.box_analytics_enabled = cfg['BOX_ANALYTICS_ENABLED']
        self.track_bundles = {}
        self.first_frame = True
        self.homography = homography
        current_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(os.path.abspath(current_path))
        background_path = os.path.join(
            parent_path, 'data', 'gui', 'maket.jpeg')
        background = cv2.imread(background_path)
        self.m_height, self.m_width, _ = background.shape

    def get_tracks(self):
        return self.tracker.tracks

    def predict(self, image, detections, db_conn):
        classes = detections['classes']
        boxes = detections['boxes']
        scores = detections['scores']

        features = self.encoder(image, boxes)
        detections = [Detection(box, score, feature, class_id) for box,
                      score, class_id, feature in zip(boxes, scores, classes, features)]

        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.oid for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(detections)
        self.update_tracker_output_v2(image, self.reid_tresh, db_conn)

    def check_point(self, x, y):
        center = np.array([x, y])
        point = np.array(center).reshape(-1, 1, 2).astype(np.float32)
        point = cv2.perspectiveTransform(point, self.homography)[0][0]
        return (0 <= point[0] <= self.m_width) and (0 <= point[1] <= self.m_height)


    def update_tracker_output_v2(self, frame, tresh, db_conn):
        self.frame_num += 1
        buffer = {}

        # Getting previous ids
        db_person_ids = self.get_distinct_person_ids(db_conn)

        # Process tracks
        for track in self.tracker.tracks:
            track_id = track.track_id
            box = track.to_tlbr().astype(int)
            xmin, ymin, xmax, ymax = box

            if self.check_point(xmin + (xmin - xmax) // 2, ymax) and \
                self.is_valid_bbox(xmin, ymin, xmax, ymax):
                image = self.crop_image(xmin, ymin, xmax, ymax, frame)
                if image.size != 0:
                    is_full_body, body_visibilities = self.is_full_body(image)
                    if is_full_body:
                        image_features = self.reidentificator.get_image_features(image)
                        buffer[track_id] = {
                            'parent_track_id': None,
                            'connections': None,
                            'features': image_features,
                            'box': box,
                            'body_visibilities': body_visibilities
                        }
                        # Check if we have connection with another track
                        parent_track_id = self.find_parent_track(track_id)
                        # If we found parent track from dictionary of connections
                        if parent_track_id:
                            buffer[track_id]['parent_track_id'] = parent_track_id
                        # There is no any connection
                        else:
                            # Getting distances with previous tracks' images
                            buffer[track_id]['connections'] = self.find_distances(tresh, db_conn, image_features, db_person_ids)

        # Process found parent tracks
        tracks = []
        blocked_parent_track_ids = []
        for track_id, value in buffer.items():
            if value['parent_track_id'] is not None:
                parent_track_id = value['parent_track_id']
                image_features = value['features']
                box = value['box']
                body_visibilities = value['body_visibilities']
                blocked_parent_track_ids.append(parent_track_id)
                tracks.append({'id': parent_track_id, 'box': box, 'body_visibilities': body_visibilities})
                self.insert_feature(parent_track_id, image_features, db_conn)

        # Process optimal connections
        results = {k: v for k, v in buffer.items() if v['parent_track_id'] is None}
        person_ids = set(db_person_ids) - set(blocked_parent_track_ids)
        blocked_track_ids = []
        for person_id in person_ids:
            person_id_connections = {}
            for track_id, value in results.items():
                if track_id not in blocked_track_ids and person_id in value['connections']:
                    person_id_connections[track_id] = value['connections'][person_id]
            if person_id_connections:
                track_id = min(person_id_connections, key=person_id_connections.get)
                blocked_track_ids.append(track_id)
                tracks.append({
                    'id': person_id, 
                    'box': results[track_id]['box'], 
                    'body_visibilities': results[track_id]['body_visibilities']
                })
                self.track_bundles[person_id].append(track_id)
                self.insert_feature(person_id, results[track_id]['features'], db_conn)

        # Process other tracks
        for track_id, value in results.items():
            if track_id not in blocked_track_ids:
                tracks.append({'id': track_id, 'box': value['box'], 'body_visibilities': value['body_visibilities']})
                self.track_bundles[track_id] = []
                self.insert_feature(track_id, value['features'], db_conn)

        # Append data to tracking output
        self.tracking_output.append({
            'frame_num': self.frame_num,
            'tracks': tracks
        })

        if self.box_analytics_enabled:
            self.process_box_analytics(tracks, frame, db_conn)

        self.first_frame = False

    def process_box_analytics(self, tracks, frame, db_conn):
        for track in tracks:
            xmin, ymin, xmax, ymax = track['box']
            image = self.crop_image(xmin, ymin, xmax, ymax, frame)
            old_images = self.get_person_images(track["id"], db_conn)
            self.insert_image(track["id"], image, db_conn)

            # Define the highlight color and thickness
            highlight_color = (0, 0, 255)
            highlight_thickness = 2

            # Create a copy of the selected image and draw a rectangle around it
            highlighted_image = image.copy()
            cv2.rectangle(highlighted_image, (0, 0), (image.shape[1], image.shape[0]), highlight_color, highlight_thickness)

            # Resize the images to a common height using cv2.resize()
            height = 300
            width = 100
            resized_images = [cv2.resize(image, (width, height)) for image in old_images]
            width = int(height * image.shape[1] / image.shape[0])
            resized_highlighted_image = cv2.resize(highlighted_image, (width, height))
            resized_images.append(resized_highlighted_image)
            hconcat1 = cv2.hconcat(resized_images)

            print(track['body_visibilities'])

            cv2.imshow(f'Player {track["id"]}', hconcat1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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
                    image_features = self.reidentificator.get_image_features(
                        image)

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
                            prev_track_id = self.find_previous_track(
                                tresh, db_conn, image_features)
                            # If a previous track ID is found
                            if prev_track_id:
                                # Just update bundle
                                self.track_bundles[prev_track_id].append(
                                    track_id)
                                tracks.append(
                                    {'id': prev_track_id, 'box': box})
                                track_id = prev_track_id
                            else:
                                tracks.append({'id': track_id, 'box': box})
                                self.track_ids.append(track_id)

                    self.insert_feature(track_id, image_features, db_conn)

        self.tracking_output.append(
            {'frame_num': self.frame_num, 'tracks': tracks})
        self.first_frame = False

    def find_parent_track(self, track_id):
        for key, values in self.track_bundles.items():
            if track_id == key or track_id in values:
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


    def find_distances(self, tresh, db_conn, image_features, db_person_ids):
        tmp_features_dict = {}

        for person_id in db_person_ids:
            features_count = self.get_person_features_count(person_id, db_conn)
            num_iterations = math.ceil(features_count / self.reid_batch_size)
            for i in range(num_iterations):
                offset = i * self.reid_batch_size
                gallery_features = self.get_person_features(
                    person_id, db_conn, self.reid_batch_size, offset)
                features = self.reidentificator.get_features_v2(
                    image_features, gallery_features)
                min_feature = features[0].min()
                if min_feature <= tresh:
                    if person_id in tmp_features_dict:
                        if min_feature < tmp_features_dict[person_id]:
                            tmp_features_dict[person_id] = min_feature
                    else:
                        tmp_features_dict[person_id] = min_feature
        
        return tmp_features_dict
    

    def find_previous_track(self, tresh, db_conn, image_features):
        db_person_ids = self.get_distinct_person_ids(db_conn)
        tmp_features_dict = self.find_distances(tresh, db_conn, image_features, db_person_ids)
        
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
                    print("WARNING: Failed to extract image patch: %s." %
                          str(box))
                    patch = np.random.uniform(
                        0., 255., image_shape).astype(np.uint8)
                image_patches.append(patch)
            image_patches = np.asarray(image_patches)
            return image_encoder(image_patches, batch_size)

        return encoder

    def get_person_features_count(self, person_id, db_conn):
        cursor = db_conn.execute(
            'SELECT COUNT(*) FROM features WHERE person_id = ?', (person_id,))
        return cursor.fetchone()[0]

    def get_person_features(self, person_id, db_conn, batch_size, offset):
        # select batch_size rows from the features table that have a matching person_id
        cursor = db_conn.execute(
            'SELECT feature FROM features WHERE person_id = ? LIMIT ? OFFSET ?', (person_id, batch_size, offset))

        # retrieve the feature vectors from the rows and convert them back to numpy arrays using pickle
        features = []
        for row in cursor:
            feature_bytes = row[0]
            feature = pickle.loads(feature_bytes)
            features.append(feature)
        features = np.stack(features)
        features = np.squeeze(features, axis=1)  # remove extra dimension

        return features
    
    def insert_image(self, person_id, image, db_conn):
        _, buffer = cv2.imencode('.jpg', image)
        image_data = buffer.tobytes()

         # insert the feature vector into the database
        db_conn.execute(
            'INSERT INTO images (person_id, image) VALUES (?, ?)', (person_id, image_data))

        # commit the transaction and close the connection
        db_conn.commit()

    def get_person_images(self, person_id, db_conn):
        images = []
        cursor = db_conn.execute(
            'SELECT image FROM images WHERE person_id = ?', (person_id,))
        for row in cursor:
            image_data = np.frombuffer(row[0], dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            images.append(image)

        return images

    def insert_feature(self, person_id, feature, db_conn):
        # convert feature vector to bytes using pickle
        feature_bytes = pickle.dumps(feature)

        # insert the feature vector into the database
        db_conn.execute(
            'INSERT INTO features (person_id, feature) VALUES (?, ?)', (person_id, feature_bytes))

        # commit the transaction and close the connection
        db_conn.commit()

    def get_distinct_person_ids(self, db_conn):
        cursor = db_conn.execute('SELECT DISTINCT person_id FROM features')
        # fetch all the results as a list of tuples
        results = cursor.fetchall()
        # extract the person_id values from the list of tuples
        person_ids = [result[0] for result in results]

        return person_ids
    
    def is_valid_bbox(self, xmin, ymin, xmax, ymax):

        w = xmax - xmin
        h = ymax - ymin

        return h / w > self.min_box_size and w >= self.min_box_width
    
    def is_full_body(self, image, shoulder_tresh=0.1, hip_tresh=0.1, nose_tresh=0.1, knee_tresh=0.75):
        pose_results = self.pose_estimator.process(image)

        if pose_results.pose_landmarks is not None:
            landmarks = pose_results.pose_landmarks.landmark
            keypoint_names = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'NOSE']
            keypoints = {name: landmarks[getattr(self.mp_pose.PoseLandmark, name)] for name in keypoint_names}

            # Check if all keypoints have sufficient visibility confidence
            if keypoints['LEFT_SHOULDER'].visibility >= shoulder_tresh and \
            keypoints['RIGHT_SHOULDER'].visibility >= shoulder_tresh and \
            keypoints['LEFT_HIP'].visibility >= hip_tresh and \
            keypoints['RIGHT_HIP'].visibility >= hip_tresh and \
            keypoints['LEFT_KNEE'].visibility >= knee_tresh and \
            keypoints['RIGHT_KNEE'].visibility >= knee_tresh and \
            keypoints['NOSE'].visibility >= nose_tresh:
                return True, {name: kp.visibility for name, kp in keypoints.items()}
            
        return False, {}
