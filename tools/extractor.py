from .texture import TextureExporter
from .helpers import (box_with_max_score, 
                      save_frame, 
                      save_homography, 
                      create_iuv, 
                      get_players_images, 
                      get_player_images_by_id,
                      get_players_images_v2,
                      get_players_images_on_frames,
                      crop_image)
import numpy as np
import os
import concurrent.futures
import cv2
import copy
from tqdm import tqdm
import imageio

class Extractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.texture_exporter = TextureExporter(cfg=cfg)
    
    def export_pitch(self, camera, homography, folder, frame_id = 0):
        player_tracks = camera.get_player_tracks()
        ball_detections = camera.get_ball_detections()
        src_frame = camera.frames[frame_id]
        mask = np.zeros((src_frame.shape[0], src_frame.shape[1], 3), np.uint8)
        for track in player_tracks[frame_id]["tracks"]:
            bbox = track['box'].astype(int)
            xmin, ymin, xmax, ymax = bbox
            mask[ymin:ymax, xmin:xmax] = 255
        ball_bbox = box_with_max_score(ball_detections[frame_id])

        if ball_bbox:
            xmin, ymin, xmax, ymax = ball_bbox
            mask[ymin:ymax, xmin:xmax] = 255
        
        frame = self.texture_exporter.get_pitch_texture_lama(src_frame, mask, homography)
        output_file = os.path.join(folder, 'pitch.jpg')
        save_frame(frame, output_file)

    def export_homography_array(self, homography_arr, homography_path):
        for i, homography in enumerate(homography_arr):
            save_homography(homography, f'{homography_path}/h_{i}')
    
    def export_analytics(self, camera_id, cameras, connections, path):
        main_camera = cameras[0]
        other_camera = cameras[camera_id]
        main_camera_player_tracks = main_camera.get_player_tracks()
        other_camera_player_tracks = other_camera.get_player_tracks()
        for main_camera_player_id in connections:
            main_player_images = get_player_images_by_id(main_camera, main_camera_player_tracks, main_camera_player_id)
            other_camera_player_id = connections[main_camera_player_id]['id']
            other_player_images = get_player_images_by_id(other_camera, other_camera_player_tracks, other_camera_player_id)
            final_image = self.stack_images(main_player_images, other_player_images)
            cv2.imwrite(f'{path}/{main_camera_player_id}_{other_camera_player_id}.jpg', final_image)

    def stack_images(self, images_array1, images_array2):
        # Determine the width and height of the final image
        widths = [img.shape[1] for img in images_array1 + images_array2]
        heights = [img.shape[0] for img in images_array1 + images_array2]
        final_width = sum(widths)
        final_height = max(heights) * 2

        # Create a black image with the final size
        final_image = np.zeros((final_height, final_width, 3), dtype=np.uint8)

        # Place images from the first array on the first row
        x_offset = 0
        for image in images_array1:
            final_image[:image.shape[0], x_offset:x_offset+image.shape[1]] = image
            x_offset += image.shape[1]

        # Place images from the second array on the second row
        x_offset = 0
        for image in images_array2:
            final_image[final_height//2:final_height//2 + image.shape[0], x_offset:x_offset+image.shape[1]] = image
            x_offset += image.shape[1]

        return final_image
    
    def export_frames(self, cameras, folder):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cameras)) as executor:
            future_to_camera = {executor.submit(self.save_frames, camera_id, camera, folder):
                                (camera_id, camera) for camera_id, camera in enumerate(cameras)}
            for future in concurrent.futures.as_completed(future_to_camera):
                camera_id, _ = future_to_camera[future]
                try:
                    future.result()
                    print(f'Camera {camera_id} processed and frames saved')
                except Exception as e:
                    print(f'Camera {camera_id} generated an exception: {e}')
                    raise Exception('Stopping app due to exception')

    def save_frames(self, camera_id, camera, frames_path):
        frame_path = os.path.join(frames_path, f'camera_{camera_id}')
        for i, frame in enumerate(camera.frames):
            save_frame(frame, f'{frame_path}/{i}.jpg')

    def set_rect_text(self, frame, bbox, text, color = (0, 0, 255), thickness = 2):
        xmin, ymin, xmax, ymax = bbox
        top_left = (xmin, ymin)
        bottom_right = (xmax, ymax)
        cv2.rectangle(frame, top_left, bottom_right, color, thickness)
        (_, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = xmin
        text_y = ymin - 10 if ymin > text_height + 10 else ymin + 10
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def export_bboxes(self, cameras, path, color = (0, 0, 255), thickness = 2):
        bboxes_path = os.path.join(path, "boxes")
        os.makedirs(bboxes_path)
        for i, camera in enumerate(cameras):
            camera_path = os.path.join(bboxes_path, f"camera_{i}")
            os.makedirs(camera_path)
            player_tracks = camera.get_player_tracks()
            ball_detections = camera.get_ball_detections()
            for j, player_track in enumerate(player_tracks):
                src_frame = camera.frames[j].copy()
                src_frame = cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB)
                for track in player_track["tracks"]:
                    bbox = track['box'].astype(int)
                    self.set_rect_text(src_frame, bbox, str(track['id']), color, thickness)
                ball_bbox = box_with_max_score(ball_detections[i])
                if ball_bbox:
                    self.set_rect_text(src_frame, ball_bbox, 'ball', color, thickness)
                cv2.imwrite(f'{camera_path}/frame_{j}.jpg', src_frame)
    
    def export_centers(self, cameras, homographies, cameras_players, path, radius = 5, color = (0, 0, 255), thickness = 2):
        centers_path = os.path.join(path, "centers")
        os.makedirs(centers_path)
        current_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(os.path.abspath(current_path))
        background_path = os.path.join(parent_path, 'data', 'gui', 'maket.jpeg')
        background = cv2.imread(background_path)
        m_height, m_width, _ = background.shape
        for i in range(len(cameras_players)):
            camera_path = os.path.join(centers_path, f"camera_{i}")
            os.makedirs(camera_path)
            centers_dict = {}
            for camera_player in cameras_players[i]:
                frame_ids = camera_player.frame_ids
                for frame_id in frame_ids:
                    if not frame_id in centers_dict:
                        centers_dict[frame_id] = []
                    center = camera_player.get_by_frame(frame_id)['h_center']
                    center = (int(center[0]), int(center[1]))
                    centers_dict[frame_id].append(center)

            for frame_id in centers_dict:
                src_frame = cameras[i].frames[frame_id].copy()
                src_frame = cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB)
                transformed_output = cv2.warpPerspective(src_frame, homographies[i], (m_width, m_height))
                for center in centers_dict[frame_id]:
                    cv2.circle(transformed_output, center, radius, color, thickness)
                cv2.imwrite(f'{camera_path}/frame_{frame_id}.jpg', transformed_output)

    def export_players_textures_v2(self, cameras, connections, texture_exporter, textures_path):
        for main_player_id in tqdm(connections.keys(), desc='Exporting images for players...'):
            player_texture = None
            if self.cfg['TEXTURES_MODE']:
                for inner_connection in connections[main_player_id]:
                    other_camera_id = inner_connection['camera_id']
                    other_player_id = inner_connection['id']
                    other_player_images = get_players_images_v2(cameras[other_camera_id], other_player_id)
                    main_player_images = get_players_images_v2(cameras[0], main_player_id)
                    player_images = other_player_images + main_player_images
                    for image in tqdm(player_images, desc=f'Combining textures for player {main_player_id}', leave=True):
                        output = texture_exporter.execute(image)[0]
                        if 'pred_densepose' in output:
                            texture = create_iuv(output, image)
                            if player_texture is None or not player_texture.any():
                                player_texture = texture
                            else:
                                player_texture = cv2.addWeighted(player_texture, 0.5, texture, 0.5, 0)
            else:
                player_texture = np.zeros((1200, 800, 3), dtype=np.uint8)
            
            # Bug fix, if we havent found detections on small video len
            if player_texture is None or not player_texture.any():
                player_texture = np.zeros((1200, 800, 3), dtype=np.uint8)

            imageio.imwrite(f'{textures_path}/player_{main_player_id}.png', player_texture)

    def export_poses(self, cameras, connections, pose_estimator):
        output = {}
        for main_player_id in tqdm(connections.keys(), desc='Exporting poses for players...'):
            output[main_player_id] = []
            for inner_connection in connections[main_player_id]:
                other_camera_id = inner_connection['camera_id']
                other_player_id = inner_connection['id']
                other_player_frames = list(inner_connection['frames'])
                other_player_images = get_players_images_on_frames(cameras[other_camera_id], other_player_id, other_player_frames)
                for i in range(len(other_player_frames)):
                    pose = pose_estimator.process(other_player_images[i])
                    output[main_player_id].append({
                        'id': main_player_id,
                        'camera_id': other_camera_id,
                        'frame': other_player_frames[i],
                        'pose': pose
                    })
        return output

    def export_players_textures(self, cameras, connections, texture_exporter, textures_path):
        players_images = get_players_images(cameras, connections)
        for player_id in tqdm(players_images, f'Processing players textures exporter'):
            player_texture = None
            if self.cfg['TEXTURES_MODE']:
                images = players_images[player_id]
                for image in tqdm(images, desc=f'Combining textures for player {player_id}', leave=True):
                    output = texture_exporter.execute(image)[0]
                    if 'pred_densepose' in output:
                        texture = create_iuv(output, image)
                        if player_texture is None or not player_texture.any():
                            player_texture = texture
                        else:
                            player_texture = cv2.addWeighted(player_texture, 0.5, texture, 0.5, 0)
            else:
                player_texture = np.zeros((1200, 800, 3), dtype=np.uint8)

            imageio.imwrite(f'{textures_path}/player_{player_id}.png', player_texture)