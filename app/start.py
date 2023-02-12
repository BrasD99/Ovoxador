import tkinter as tk
from .pages import (VideoProcessPage, UploadPage)

class App(tk.Tk):

    def __init__(self, *args, **kwargs):
		
        tk.Tk.__init__(self, *args, **kwargs)
		
        self.title('ML Soccer Analysis')
        self.resizable(False, False)

        container = tk.Frame(self)
        container.pack(side = "top", fill = "both", expand = True)

        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)

        self.frames = {}

        for F in (UploadPage, VideoProcessPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row = 0, column = 0, sticky ="nsew")
        self.show_frame(UploadPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        #If calling VideoProcessPage, refreshing page
        if cont == VideoProcessPage:
            frame.refresh(self.single_video_path, self.multiple_video_path)
    
    def set_videos(self, single_video_path, multiple_video_path):
        self.single_video_path = single_video_path
        self.multiple_video_path = multiple_video_path