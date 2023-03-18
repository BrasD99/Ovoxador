import tkinter as tk
from PIL import Image, ImageTk


class ImageWithPoints(tk.Canvas):
    def __init__(
            self,
            parent,
            image_index,
            src_image_index,
            pil_image,
            callback,
            show_points=True,
            set_camera_point=False):
        super().__init__(parent, bg='green')
        self.parent = parent
        self.image_index = image_index
        self.set_camera_point = set_camera_point
        self.src_image_index = src_image_index
        self.pil_image = pil_image
        self.show_points = show_points
        self.original_pil_size = {
            'height': pil_image.height, 'width': pil_image.width}
        self.callback = callback
        self.points = []
        self.point_colors = ['green', 'blue', 'red', 'yellow', 'white']
        self.point_coords = []
        self.bind("<Configure>", self.resize_image)
        self.bind("<ButtonRelease-1>", self.button_released)

    def get_original_points(self):
        scale_height = self.original_pil_size['height'] / self.pil_image.height
        scale_width = self.original_pil_size['width'] / self.pil_image.width

        return [
            {'x': point['x'] * scale_width, 'y': point['y'] * scale_height}
            for point in self.point_coords
        ]

    def do_callback(self):
        original_points = self.get_original_points()
        self.callback(self.image_index, self.src_image_index,
                      original_points, self.point_coords)

    def button_released(self, event):
        if (self.show_points):
            self.do_callback()

    def update_image(self, pil_image, points=[], reset_points=False):
        self.original_pil_size = {
            'height': pil_image.height, 'width': pil_image.width}
        self.pil_image = pil_image.resize(
            (self.winfo_width(), self.winfo_height()), Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(self.pil_image)
        self.create_image(0, 0, image=self.image, anchor='nw', tags="image")

        if reset_points:
            self.points = []
            self.point_coords = []
        elif points:
            self.point_coords = points
        self.replace_points()

    def replace_points(self):
        points_count = 4
        if self.set_camera_point:
            points_count = 5

        if self.show_points:
            new_points = [
                {'x': 20 + i * 20, 'y': 20 + i * 20}
                for i in range(points_count)
            ] if len(self.points) == 0 else self.point_coords

            self.point_coords = []
            self.points = []

            for i, point in enumerate(new_points):
                self.add_point(point['x'], point['y'], i)

    def clear(self):
        self.delete('all')

    def resize_image(self, event):
        self.delete("image")
        new_width = event.width
        new_height = event.height
        self.pil_image = self.pil_image.resize(
            (new_width, new_height), Image.LANCZOS)
        self.image = ImageTk.PhotoImage(self.pil_image)
        self.create_image(0, 0, image=self.image, anchor='nw', tags="image")
        self.replace_points()

    def add_point(self, x, y, i):
        point = self.create_rectangle(
            x-5, y-5, x+5, y+5, fill=self.point_colors[i], tags=i)
        self.points.append(point)
        self.point_coords.append({'x': x, 'y': y})
        self.tag_bind(point, "<B1-Motion>",
                      lambda event: self.move_point(event, point))

    def move_point(self, event, point):
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        x = max(10, min(canvas_width - 10, event.x))
        y = max(10, min(canvas_height - 10, event.y))
        self.coords(point, x - 5, y - 5, x + 5, y + 5)
        tag = int(self.gettags(point)[0])
        self.point_coords[tag] = {'x': x, 'y': y}
