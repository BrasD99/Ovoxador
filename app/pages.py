import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from .canvas.image_with_points import ImageWithPoints
import os

class UploadPage(tk.Frame):
    def check_uploaded(self):
        if self.single_video_uploaded and self.multiple_videos_uploaded:
            self.confirm_btn.grid(row = 4, column = 0, columnspan = 2, sticky='nesw')
        else:
            self.confirm_btn.grid_forget()

    def upload_single_video(self):
        self.single_video = filedialog.askopenfilename(title="Choose a video", filetypes=[("Video Files", "*.mov")])
        self.single_video_label.config(text="Single video uploaded")
        self.single_video_label.config(fg="green")
        self.single_video_uploaded = True
        self.check_uploaded()

    def upload_multiple_videos(self):
        self.multiple_videos = filedialog.askopenfilenames(title="Choose videos", filetypes=[("Video Files", "*.mov")])
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

        self.confirm_btn = ttk.Button(self, text ="Confirm",
                                    command = lambda : self.on_confirm_click(controller))

        single_video_button = ttk.Button(self, text="Upload Single Video", command=self.upload_single_video)
        single_video_button.grid(row = 0, column = 0, columnspan = 2, sticky='nesw')

        self.single_video_label = tk.Label(self, text="Single video not uploaded")
        self.single_video_label.grid(row = 1, column = 0, columnspan = 2, sticky='nesw')

        multiple_videos_button = ttk.Button(self, text="Upload Multiple Videos", command=self.upload_multiple_videos)
        multiple_videos_button.grid(row = 2, column = 0, columnspan = 2, sticky='nesw')

        self.multiple_videos_label = tk.Label(self, text="Multiple videos not uploaded")
        self.multiple_videos_label.grid(row = 3, column = 0, columnspan = 2, sticky='nesw')

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
            image_index = 0, 
            src_image_index=0, 
            pil_image = self.unselected_img, 
            callback = self.set_points,
            show_points = False)

        self.left_image.grid(row=0, column=0, sticky='nesw')

        dst_img_file = os.path.join(image_path, 'maket.jpeg')
        self.dst_img = self.read_cv_image(dst_img_file)
        self.dst_pil_img = self.cv_to_pil(self.dst_img)

        self.right_image = ImageWithPoints(
            self, 
            image_index = 1, 
            src_image_index=0, 
            pil_image = self.dst_pil_img, 
            callback = self.set_points,
            show_points = False)
        
        self.right_image.grid(row=0, column=1, sticky='nesw')

        back_btn = ttk.Button(self, text = "Back", command = lambda : controller.show_frame(UploadPage))
        back_btn.grid(row = 3, column = 0, columnspan = 1)

        next_btn = ttk.Button(self, text = "Next", command = lambda : self.save_homography())
        next_btn.grid(row = 3, column = 1, columnspan = 1)

        self.canvas = tk.Canvas(self, width=800, height=150)
        self.canvas.grid(row=1, column=0, columnspan=2)

        scrollbar = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
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

    def save_homography(self):
        for video_id in range(len(self.image_points)):
            if self.image_points[video_id] and self.dst_points[video_id]:
                src_points = np.array([[point['x'], point['y']] for point in self.image_points[video_id]], dtype=np.float32)
                dst_points = np.array([[point['x'], point['y']] for point in self.dst_points[video_id]], dtype=np.float32)
                homography_matrix, _ = cv2.findHomography(src_points, dst_points)
                filename = f"/Users/brasd99/Downloads/homography_{video_id}.yml"
                cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE).write("homography", homography_matrix)

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
        src_points = np.array([[point['x'], point['y']] for point in self.image_points[frame_index]], dtype=np.float32)
        dst_points = np.array([[point['x'], point['y']] for point in self.dst_points[frame_index]], dtype=np.float32)
        homography_matrix, _ = cv2.findHomography(src_points, dst_points)
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

    def set_text(self, frame, text, font_scale = 5, thickness = 10):
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        h, w, _ = frame.shape
        text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = int((w - text_width) / 2)
        text_y = int((h + text_height) / 2)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
    
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
            self.right_image.update_image(pil, points = self.dst_orig_points[frame_id])
        else:
            self.right_image.update_image(self.dst_pil_img, reset_points = True)
        
        self.left_image.show_points = True
        self.left_image.update_image(image, reset_points = len(orig_points) == 0, points = orig_points)

    def set_label(self, video_path, text, offset = 0):
        frame, orig_frame = self.get_first_frame(video_path, text)
        self.frames.append(frame)
        self.orig_frames.append(orig_frame)
        self.image_points.append([])
        self.orig_points.append([])
        self.dst_points.append([])
        self.dst_orig_points.append([])
        orig_frame = self.cv_to_pil(orig_frame)
        self.pil_images.append(orig_frame)
        image = self.to_photo_image(frame, desired_height = self.canvas.winfo_height() - 10)
        self.photo_images.append(image)

        label = tk.Label(self.canvas, image=image)
        self.labels.append(label)

        label.bind("<Button-1>", self.on_label_click)

        offset += image.width() + 5 if offset else image.width() / 2 + 5
        
        self.canvas.create_window(offset, image.height() / 2 + 5, window=label, anchor=tk.CENTER)

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
            offset = self.set_label(video_path, f'Camera {i + 1}', offset = offset)