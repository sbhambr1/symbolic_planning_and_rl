import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import time


_info_displayer_main_window = None
_info_displayer_canvas = None


def sleep(secs):
    global _info_displayer_main_window
    if _info_displayer_main_window is None:
        if secs > 0:
            time.sleep(secs)
    else:
        _info_displayer_main_window.update_idletasks()
        _info_displayer_main_window.after(int(1000 * secs), _info_displayer_main_window.quit)
        _info_displayer_main_window.mainloop()


# noinspection DuplicatedCode
class InfoDisplayer:
    def __init__(self, screen_width, screen_height, frame_time=0.1):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.frame_time = frame_time
        self.img_buffer = []

    def display_img_gray(self, gray_img, location=(0, 0)):
        assert len(gray_img.shape) == 2, 'display_img_grey expect 2d grayscale image but got: ' + str(gray_img.shape)

        global _info_displayer_canvas
        if _info_displayer_canvas is None:
            self._init_window()

        img = ImageTk.PhotoImage(master=_info_displayer_canvas, image=Image.fromarray(gray_img.astype(np.uint8), 'L'))
        self.img_buffer.append(img)
        _info_displayer_canvas.create_image(location, image=img, anchor="nw")

    def display_img_rgb(self, rgb_img, location=(0, 0)):
        assert len(rgb_img.shape) == 3, 'display_img_rgb expect 3-channel rgb image but got: ' + str(rgb_img.shape)

        global _info_displayer_canvas
        if _info_displayer_canvas is None:
            self._init_window()

        img = ImageTk.PhotoImage(master=_info_displayer_canvas, image=Image.fromarray(rgb_img.astype(np.uint8), 'RGB'))
        self.img_buffer.append(img)
        _info_displayer_canvas.create_image(location, image=img, anchor="nw")

    def refresh(self):
        global _info_displayer_main_window
        if _info_displayer_main_window is None:
            self._init_window()
        sleep(self.frame_time)
        self.img_buffer.clear()

    def _init_window(self):
        global _info_displayer_main_window, _info_displayer_canvas

        if _info_displayer_main_window is None:
            _info_displayer_main_window = tk.Tk()
            _info_displayer_main_window.geometry('%dx%d+%d+%d' % (self.screen_width, self.screen_height, 10, 0))
            _info_displayer_main_window.resizable(False, False)

            _info_displayer_canvas = tk.Canvas(_info_displayer_main_window,
                                               width=self.screen_width - 1,
                                               height=self.screen_height - 1)
            _info_displayer_canvas.pack()



