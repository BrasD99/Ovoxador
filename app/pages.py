import tqdm
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from .canvas.image_with_points import ImageWithPoints
import os
from tools.helpers import (
    get_config,
    get_cameras_config,
    create_output_directory)
from tools.camera import Camera
from tools.extractor import Extractor, TextureExporter
from tools.pose import PoseEstimator
import shutil
import threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class UploadPage(tk.Frame):
    def check_uploaded(self):
        if self.single_video_uploaded and self.multiple_videos_uploaded:
            self.confirm_btn.grid(row=4, column=0, columnspan=2, sticky='nesw')
        else:
            self.confirm_btn.grid_forget()

    def upload_single_video(self):
        self.single_video = filedialog.askopenfilename(
            title="Choose a video", filetypes=[("Video Files", "*.mov")])
        self.single_video_label.config(text="Single video uploaded")
        self.single_video_label.config(fg="green")
        self.single_video_uploaded = True
        self.check_uploaded()

    def upload_multiple_videos(self):
        self.multiple_videos = filedialog.askopenfilenames(
            title="Choose videos", filetypes=[("Video Files", "*.mov")])
        self.multiple_videos_label.config(text="Multiple videos uploaded")
        self.multiple_videos_label.config(fg="green")
        self.multiple_videos_uploaded = True
        self.check_uploaded()

    def on_confirm_click(self, controller):
        controller.set_videos(self.single_video, self.multiple_videos)
        controller.show_frame(VideoProcessPage)

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.single_video_uploaded = False
        self.multiple_videos_uploaded = False

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.confirm_btn = ttk.Button(self, text="Confirm",
                                      command=lambda: self.on_confirm_click(controller))

        single_video_button = ttk.Button(
            self, text="Upload Main camera Video", command=self.upload_single_video)
        single_video_button.grid(row=0, column=0, columnspan=2, sticky='nesw')

        self.single_video_label = tk.Label(
            self, text="Main camera video not uploaded")
        self.single_video_label.grid(
            row=1, column=0, columnspan=2, sticky='nesw')

        multiple_videos_button = ttk.Button(
            self, text="Upload Other cameras Videos", command=self.upload_multiple_videos)
        multiple_videos_button.grid(
            row=2, column=0, columnspan=2, sticky='nesw')

        self.multiple_videos_label = tk.Label(
            self, text="Other cameras videos not uploaded")
        self.multiple_videos_label.grid(
            row=3, column=0, columnspan=2, sticky='nesw')


class VideoProcessPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.image_points = []
        self.orig_points = []
        self.dst_points = []
        self.dst_orig_points = []
        self.pil_images = []
        self.photo_images = []

        current_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(os.path.abspath(current_path))
        image_path = os.path.join(parent_path, 'data', 'gui')

        self.unselected_img_file = os.path.join(image_path, 'default.jpeg')
        self.unselected_img = self.read_cv_image(self.unselected_img_file)
        self.unselected_img = self.cv_to_pil(self.unselected_img)

        self.left_image = ImageWithPoints(
            self,
            image_index=0,
            src_image_index=0,
            pil_image=self.unselected_img,
            callback=self.set_points,
            show_points=False,
            set_camera_point=False)

        self.left_image.grid(row=0, column=0, sticky='nesw')

        dst_img_file = os.path.join(image_path, 'maket.jpeg')
        self.dst_img = self.read_cv_image(dst_img_file)
        self.dst_pil_img = self.cv_to_pil(self.dst_img)

        self.right_image = ImageWithPoints(
            self,
            image_index=1,
            src_image_index=0,
            pil_image=self.dst_pil_img,
            callback=self.set_points,
            show_points=False,
            set_camera_point=True)

        self.right_image.grid(row=0, column=1, sticky='nesw')

        back_btn = ttk.Button(
            self, text="Back", command=lambda: controller.show_frame(UploadPage))
        back_btn.grid(row=3, column=0, columnspan=1)

        self.next_btn = ttk.Button(
            self, text="Next", command=lambda: self.confirm_homography(controller))
        self.next_btn.grid(row=3, column=1, columnspan=1)

        self.canvas = tk.Canvas(self, width=800, height=150)
        self.canvas.grid(row=1, column=0, columnspan=2)

        scrollbar = tk.Scrollbar(
            self, orient="horizontal", command=self.canvas.xview)
        scrollbar.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.canvas.configure(xscrollcommand=scrollbar.set)

        self.frames = []
        self.orig_frames = []
        self.labels = []

    def cv_to_pil(self, image):
        return Image.fromarray(np.uint8(image))

    def pil_to_tk(self, image):
        return ImageTk.PhotoImage(image)

    def read_cv_image(self, image_path):
        img = cv2.imread(image_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def confirm_homography(self, controller):
        src_points_dict = {}
        dst_points_dict = {}

        for video_id in range(len(self.image_points)):
            if self.image_points[video_id] and self.dst_points[video_id]:
                src_points = np.array(
                    [[point['x'], point['y']] for point in self.image_points[video_id]], dtype=np.float32)
                dst_points = np.array(
                    [[point['x'], point['y']] for point in self.dst_points[video_id]], dtype=np.float32)
                src_points_dict[video_id] = src_points
                dst_points_dict[video_id] = dst_points

        controller.set_points(src_points_dict, dst_points_dict)
        controller.show_frame(Processor)

    def save_homography(self):
        for video_id in range(len(self.image_points)):
            if self.image_points[video_id] and self.dst_points[video_id]:
                src_points = np.array(
                    [[point['x'], point['y']] for point in self.image_points[video_id]], dtype=np.float32)
                dst_points = np.array(
                    [[point['x'], point['y']] for point in self.dst_points[video_id]], dtype=np.float32)
                homography_matrix, _ = cv2.findHomography(
                    src_points, dst_points)
                filename = f"/Users/brasd99/Downloads/homography_{video_id}.yml"
                cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE).write(
                    "homography", homography_matrix)

    def set_points(self, image_index, src_image_index, points, orig_points):
        if image_index == 0:
            self.image_points[src_image_index] = points
            self.orig_points[src_image_index] = orig_points
        else:
            self.dst_points[src_image_index] = points
            self.dst_orig_points[src_image_index] = orig_points

        if self.image_points[src_image_index] and self.dst_points[src_image_index]:
            pil_img = self.calculate_homography(src_image_index)
            self.right_image.update_image(pil_img)

    def calculate_homography(self, frame_index):
        frame = self.orig_frames[frame_index]
        src_points = np.array([[point['x'], point['y']]
                              for point in self.image_points[frame_index]], dtype=np.float32)
        dst_points = np.array([[point['x'], point['y']]
                              for point in self.dst_points[frame_index]], dtype=np.float32)
        homography_matrix, _ = cv2.findHomography(src_points, dst_points[:-1])
        rows, cols = self.dst_img.shape[:2]
        result = cv2.warpPerspective(frame, homography_matrix, (cols, rows))
        result = cv2.addWeighted(self.dst_img, 0.5, result, 0.5, 0)
        return Image.fromarray(np.uint8(result))

    def to_photo_image(self, image, desired_height):
        height, width = image.shape[:2]
        aspect_ratio = width / height

        desired_width = int(desired_height * aspect_ratio)

        image = Image.fromarray(np.uint8(image))
        image = image.resize((desired_width, desired_height), Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    def set_text(self, frame, text, font_scale=5, thickness=10):
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        h, w, _ = frame.shape
        text_width, text_height = cv2.getTextSize(
            text, font, font_scale, thickness)[0]
        text_x = int((w - text_width) / 2)
        text_y = int((h + text_height) / 2)
        cv2.putText(frame, text, (text_x, text_y),
                    font, font_scale, color, thickness)

    def get_first_frame(self, video_path, text):
        cap = cv2.VideoCapture(video_path)

        _, frame = cap.read()
        orig_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        processed_frame = orig_frame.copy()

        self.set_text(processed_frame, text)

        cap.release()

        return processed_frame, orig_frame

    def on_label_click(self, event):
        for label in self.labels:
            label.config(bg='white')
        event.widget.config(bg='blue')
        frame_id = self.labels.index(event.widget)
        orig_points = self.orig_points[frame_id]
        image = self.pil_images[frame_id]

        self.right_image.src_image_index = frame_id
        self.left_image.src_image_index = frame_id

        self.right_image.show_points = True

        if len(self.image_points[frame_id]) > 0 and len(self.dst_points[frame_id]):
            pil = self.calculate_homography(frame_id)
            self.right_image.update_image(
                pil, points=self.dst_orig_points[frame_id])
        else:
            self.right_image.update_image(self.dst_pil_img, reset_points=True)

        self.left_image.show_points = True
        self.left_image.update_image(image, reset_points=len(
            orig_points) == 0, points=orig_points)

    def set_label(self, video_path, text, offset=0):
        frame, orig_frame = self.get_first_frame(video_path, text)
        self.frames.append(frame)
        self.orig_frames.append(orig_frame)
        self.image_points.append([])
        self.orig_points.append([])
        self.dst_points.append([])
        self.dst_orig_points.append([])
        orig_frame = self.cv_to_pil(orig_frame)
        self.pil_images.append(orig_frame)
        image = self.to_photo_image(
            frame, desired_height=self.canvas.winfo_height() - 10)
        self.photo_images.append(image)

        label = tk.Label(self.canvas, image=image)
        self.labels.append(label)

        label.bind("<Button-1>", self.on_label_click)

        offset += image.width() + 5 if offset else image.width() / 2 + 5

        self.canvas.create_window(
            offset, image.height() / 2 + 5, window=label, anchor=tk.CENTER)

        return offset

    def reset(self):
        self.image_points = []
        self.orig_points = []
        self.dst_points = []
        self.dst_orig_points = []
        self.pil_images = []
        self.photo_images = []

        self.left_image.clear()
        self.right_image.clear()

        self.left_image.show_points = False
        self.left_image.update_image(self.unselected_img, reset_points=True)

        self.right_image.show_points = False
        self.right_image.update_image(self.dst_pil_img, reset_points=True)

        for label in self.labels:
            label.destroy()
        self.labels.clear()
        self.frames.clear()

    def refresh(self, single_video_path, multiple_videos_path):
        self.single_video_path = single_video_path
        self.multiple_videos_path = multiple_videos_path

        self.reset()

        offset = self.set_label(single_video_path, 'Main Camera')

        for i, video_path in enumerate(list(multiple_videos_path)):
            offset = self.set_label(
                video_path, f'Camera {i + 1}', offset=offset)


class Processor(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Create a canvas widget to hold the frame
        canvas = tk.Canvas(self)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar to the canvas
        scrollbar = tk.Scrollbar(
            self, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Attach the scrollbar to the canvas
        canvas.config(yscrollcommand=scrollbar.set)

        # Create a frame to hold the labels and entry widgets
        frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor=tk.NW)

        cameras_count = 3

        self.config = get_config()
        self.cameras_cfg = get_cameras_config(
            self.config, [i for i in range(cameras_count)])

        row_num = 0

        blocked_keys = ['DEFAULT_CAMERA_PARAMS', 'CAMERA_OVERRIDEN_PARAMS']

        main_config_label = tk.Label(
            frame, text='Main config', font=("TkDefaultFont", 20, "bold"))
        main_config_label.grid(column=0, row=row_num, columnspan=2, sticky="w")

        row_num += 1

        for key, value in self.config.items():
            if not key in blocked_keys:
                # Create a label for the parameter
                label = tk.Label(frame, text=key)
                label.grid(column=0, row=row_num)

                # Create an entry widget for the parameter
                var = tk.StringVar(frame, value=str(value))
                entry = tk.Entry(frame, textvariable=var)

                entry.grid(column=1, row=row_num, columnspan=2, sticky="w")
                entry.bind("<FocusOut>", lambda event, key=key,
                           var=var: self.update_config(self.config, key, var))
                row_num += 1

        for id in self.cameras_cfg:
            camera_name = f'Camera {id + 1}'
            if id == 0:
                camera_name = 'Main camera'

            camera_label = tk.Label(
                frame, text=camera_name, font=("TkDefaultFont", 20, "bold"))
            camera_label.grid(column=0, row=row_num, columnspan=2, sticky="w")
            row_num += 1

            for key, value in self.cameras_cfg[id].items():
                if not key in self.config:
                    # Create a label for the parameter
                    label = tk.Label(frame, text=key)
                    label.grid(column=0, row=row_num)

                    # Create an entry widget for the parameter
                    var = tk.StringVar(frame, value=str(value))
                    entry = tk.Entry(frame, textvariable=var)

                    entry.grid(column=1, row=row_num)
                    entry.bind("<FocusOut>", lambda event, key=key, var=var: self.update_config(
                        self.cameras_cfg[id], key, var))

                    row_num += 1

        confirm_btn = ttk.Button(
            frame, text="Confirm", command=lambda: self.confirm_clicked(canvas, frame, scrollbar))
        confirm_btn.grid(row=row_num, column=0, columnspan=1)
        # Update the scroll region of the canvas to fit the size of the frame
        frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

    def confirm_clicked(self, canvas, frame, scrollbar):
        canvas.yview_moveto(0)
        scrollbar.pack_forget()

        # Hide all labels and entries
        for child in frame.winfo_children():
            child.grid_forget()

        progress_var = tk.DoubleVar()
        progress_bar = tk.ttk.Progressbar(
            frame, variable=progress_var, maximum=100)
        progress_bar.grid(column=0, row=0, columnspan=2)

        my_thread = threading.Thread(target=self.start, args=(progress_var,))
        my_thread.start()

    def update_config(self, config, key, value):
        config[key] = value.get()

    def update_entry(self):
        # get the selected key
        selected_key = self.key_listbox.get(self.key_listbox.curselection())

        # update the entry widget with the value of the selected key
        if selected_key in self.config:
            self.value_entry.delete(0, tk.END)
            self.value_entry.insert(0, str(self.config[selected_key]))

    def refresh(self, src_points_dict, dst_points_dict, single_video_path, multiple_video_path):
        self.src_points_dict = src_points_dict
        self.dst_points_dict = dst_points_dict
        self.single_video_path = single_video_path
        self.multiple_video_path = multiple_video_path

    def start(self, progress_var):
        homographies, cameras_locations = self.get_homographies_and_cameras()
        cameras_locations = [cameras_location.tolist() for cameras_location in cameras_locations]
        video_paths = self.get_video_paths()
        output_src_dict = create_output_directory(self.config['OUTPUT_FOLDER'], video_paths)

        cameras = [
            Camera(
                video_file=video_paths[i],
                camera_id=i,
                homography=homographies[i],
                cfg=self.merge_configs(self.cameras_cfg[i])
            )
            for i in range(len(video_paths))
        ]

        for i, camera in enumerate(cameras):
            camera.process()
            percent = (i + 1) * (100 / len(cameras))
            progress_var.set(percent)

        extractor = Extractor(cfg=self.config)
        texture_exporter = TextureExporter(cfg=self.config)
        pose_estimator = PoseEstimator(cfg=self.config)

        export_dict = {
            'boxes': True,
            'centers': True,
            'dump': True,
            'frames': True,
            'homography': True,
            'pitch_texture': True,
            'players_texture': True
        }

        extractor.export_all(
            cameras=cameras,
            cameras_locations=cameras_locations,
            homographies=homographies,
            texture_exporter=texture_exporter,
            pose_estimator=pose_estimator,
            export_dict=export_dict,
            output_src_dict=output_src_dict,
            progress_var=progress_var)

    def merge_configs(self, camera_config):
        for key in self.config.keys():
            if key in camera_config:
                camera_config[key] = self.config[key]
        return camera_config

    def get_video_paths(self):
        return [self.single_video_path] + list(self.multiple_video_path)

    def get_homographies_and_cameras(self):
        homography_arr = []
        cameras_location_arr = []
        for video_id in self.src_points_dict.keys():
            src_points = self.src_points_dict[video_id]
            dst_points = self.dst_points_dict[video_id]
            homography_matrix, _ = cv2.findHomography(
                src_points, dst_points[:-1])
            homography_arr.append(homography_matrix)
            cameras_location_arr.append(dst_points[len(dst_points) - 1])
        return homography_arr, cameras_location_arr
