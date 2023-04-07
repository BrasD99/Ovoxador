from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm

from .detector import Detector
from .helpers import filter_detections, print_warning
from .tracker import Tracker

import sqlite3
import os


class Camera:
    def __init__(self, video_file, camera_id, homography, cfg):
        self.fps = cfg["CAMERA_FPS"]
        self.max_video_len = cfg['MAX_VIDEO_LENGTH']
        self.video_file = video_file
        self.camera_id = camera_id
        self.classes = {'player': 0, 'ball': 32}
        self.detector = Detector(
            player_prob_tresh=cfg['PLAYER_DETECTION_TRESH'],
            ball_prob_tresh=cfg['BALL_DETECTION_TRESH'],
            class_ids=list(self.classes.values()),
            use_latest=cfg['USE_YOLOv8'])
        self.player_tracker = Tracker(
            cfg, reID_model_path=cfg['REID_MODEL_PATH'], homography=homography, nms_max_overlap=cfg['MAX_BBOX_OVERLAP'])
        self.ball_detections = []
        self.frames = []
        self.homography = homography

    def process(self):
        original_clip = VideoFileClip(self.video_file)
        original_fps = original_clip.fps

        if original_fps < self.fps:
            print_warning(
                f'video fps: {original_fps} is less then fps from config: {float(self.fps)}')

        processed_clip = original_clip.set_fps(self.fps)

        if self.max_video_len > 0:
            processed_clip = processed_clip.subclip(0, self.max_video_len)

        frames_count = int(processed_clip.fps * processed_clip.duration)

        desc = f'Processing camera №{self.camera_id} [frames≈{frames_count + 1}]'
        if self.camera_id == 0:
            desc = f'Processing main camera [frames≈{frames_count}]'

        # Creating temporary db to store camera data
        db_file_name = f'tmp_{self.camera_id}.db'
        conn = self.create_db(db_file_name)

        for frame in tqdm(processed_clip.iter_frames(), desc=desc):
            self.frames.append(frame)

            detections = self.detector.predict(frame)

            ball_detections = filter_detections(
                detections, self.classes['ball'])
            player_detections = filter_detections(
                detections, self.classes['player'])

            self.player_tracker.predict(frame, player_detections, conn)
            self.ball_detections.append(ball_detections)

        self.remove_db(conn, db_file_name)

    def create_db(self, file_name):
        if os.path.exists(file_name):
            os.remove(file_name)
        conn = sqlite3.connect(file_name)
        # create a table to store the image features
        conn.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY,
                person_id INTEGER,
                feature BLOB
            )
        ''')
        # create a table to store the images for analytics
        conn.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                person_id INTEGER,
                image BLOB
            )
        ''')
        return conn

    def remove_db(self, conn, file_name):
        conn.close()
        os.remove(file_name)

    def get_player_tracks(self):
        return self.player_tracker.tracking_output

    def get_ball_detections(self):
        return self.ball_detections

    def get_frames_num(self):
        return len(self.frames)
