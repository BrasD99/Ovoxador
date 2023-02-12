'''
import cv2
import imageio
import numpy as np
import torch
from setuptools import find_packages
from tools.camera import Camera
from tools.helpers import create_iuv, get_config
from tools.pose import PoseEstimator
from tools.texture import TextureExporter
from tqdm import tqdm
'''
from app.start import App


app = App()
app.mainloop()


'''
packages = find_packages()
print(packages)

cfg = get_config()

camera = Camera(
    video_file='/Users/brasd99/Downloads/IMG_4848.mov', 
    camera_id=1, 
    cfg=cfg)

camera.process()

'''
'''
pose_estimator = PoseEstimator()

for i, player_track in enumerate(player_tracks):
    for j in tqdm(range(len(player_track["tracks"])), f'Processing frame {i + 1}', leave=True):
        bbox = player_track['tracks'][j]['box'].astype(int)
        xmin, ymin, xmax, ymax = bbox
        player_image = camera.frames[i][ymin:ymax, xmin:xmax]
        result = pose_estimator.process(player_image).pose_world_landmarks.landmark
        print(result[0].x) #x, y, z, visibility




player_tracks, ball_tracks = camera.get_tracks()

color = (0, 0, 255)  # Red
thickness = 2

for i, player_track in enumerate(player_tracks):
    image = cv2.cvtColor(camera.frames[i], cv2.COLOR_BGR2RGB)
    for track in player_track["tracks"]:
        bbox = track['box'].astype(int)
        xmin, ymin, xmax, ymax = bbox
        top_left = (xmin, ymin)
        bottom_right = (xmax, ymax)
        cv2.rectangle(image, top_left, bottom_right, color, thickness)
        (text_width, text_height) = cv2.getTextSize(str(track['id']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = xmin
        text_y = ymin - 10 if ymin > text_height + 10 else ymin + 10
        cv2.putText(image, str(track['id']), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imwrite(f'/Users/brasd99/Desktop/Dissertation/outputs/detections/frame{i}.jpg', image)

texture_exporter = TextureExporter(cfg=cfg)

textures = {}

for i in tqdm(range(len(player_tracks)), f'Processing frames', leave=False):
    player_track = player_tracks[i]
    image = camera.frames[i]
    for j in tqdm(range(len(player_track["tracks"])), f'Processing frame {i + 1}', leave=True):
        if cfg['TEXTURES_MODE'] == 1:
            bbox = player_track['tracks'][j]['box'].astype(int)
            xmin, ymin, xmax, ymax = bbox
            player_image = image[ymin:ymax, xmin:xmax]
            output = texture_exporter.execute(player_image)[0]
            if 'pred_densepose' in output:
                texture = create_iuv(output, player_image)
                if not player_track['tracks'][j]['id'] in textures:
                    textures.update({player_track['tracks'][j]['id']: texture})
                else:
                    textures[player_track['tracks'][j]['id']] = cv2.addWeighted(textures[player_track['tracks'][j]['id']], 0.5, texture, 0.5, 0)
        else:
            if not player_track['tracks'][j]['id'] in textures:
                empty_image = np.zeros((1200, 800, 3), dtype=np.uint8)
                textures.update({player_track['tracks'][j]['id']: empty_image})




for i, (key, value) in enumerate(textures.items()):
    imageio.imwrite(f'/Users/brasd99/Desktop/Dissertation/outputs/texture{i + 1}.jpg', value)
'''