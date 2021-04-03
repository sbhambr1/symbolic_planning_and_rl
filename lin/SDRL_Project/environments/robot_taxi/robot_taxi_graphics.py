import time

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from skimage.draw import disk, rectangle, line

DEFAULT_GRID_SIZE = 50
Passenger_SIZE = 0.36
AGENT_SIZE = 0.36
GRID_LINE_COLOR = (137, 137, 137)

_robot_taxi_main_window = None
_taxi_robot_canvas = None


# noinspection PyUnresolvedReferences
def robot_taxi_graphics_sleep(secs):
    global _robot_taxi_main_window
    if _robot_taxi_main_window is None:
        time.sleep(secs)
    else:
        _robot_taxi_main_window.update_idletasks()
        _robot_taxi_main_window.after(int(1000 * secs), _robot_taxi_main_window.quit)
        _robot_taxi_main_window.mainloop()


class Robot_Taxi_Graphics:
    def __init__(self, width, height, zoom=1.0, frame_time=0.0, is_render=False):
        self.zoom = zoom
        self.frame_time = frame_time
        self.grid_size = float(int(DEFAULT_GRID_SIZE*zoom))

        self.screen_width = int(self.grid_size*width)
        self.screen_height = int(self.grid_size * height)

        self.background_img = np.ones(shape=(self.screen_height, self.screen_width, 3)).astype(np.uint8) * 255
        self._draw_background(height, width)
        self.rgb_observation = self.background_img.copy()

    def _draw_background(self, height, width):
        # horizontal lines
        for h in range(1, height):
            pos_x = int(h*self.grid_size)
            rr, cc = line(pos_x, 0, pos_x, self.screen_width-1)
            self.background_img[rr, cc, :] = list(GRID_LINE_COLOR)
        # vertical lines
        for w in range(1, width):
            pos_y = int(w*self.grid_size)
            rr, cc = line(0, pos_y, self.screen_height-1, pos_y)
            self.background_img[rr, cc, :] = list(GRID_LINE_COLOR)

    def _get_screen_pos(self, pos_x, pos_y):
        return pos_x * self.grid_size, pos_y * self.grid_size

    def _draw_passengers(self, location, color):
        passenger_radius = Passenger_SIZE * self.grid_size

        pos_x, pos_y = self._get_screen_pos(location[0], location[1])
        pos_x += int(self.grid_size / 2)
        pos_y += int(self.grid_size / 2)
        rr, cc = disk((pos_x, pos_y), passenger_radius)
        self.rgb_observation[rr, cc, :] = list(color)

    def _draw_agent(self, location, color):
        agent_radius = AGENT_SIZE * self.grid_size
        pos_x, pos_y = self._get_screen_pos(location[0], location[1])
        # move to grid center
        pos_x += int(self.grid_size/2)
        pos_y += int(self.grid_size/2)

        rr, cc = disk((pos_x, pos_y), agent_radius)
        self.rgb_observation[rr, cc, :] = list(color)

    def _draw_dest(self, location, color):
        pos_x, pos_y = self._get_screen_pos(location[0], location[1])
        end_x = pos_x + self.grid_size - 1  # -1 to prevent out of bound
        end_y = pos_y + self.grid_size - 1  # -1 to prevent out of bound

        rr, cc = rectangle(start=(pos_x, pos_y), end=(end_x, end_y))
        rr = rr.astype(np.int32)
        cc = cc.astype(np.int32)
        self.rgb_observation[rr, cc, :] = list(color)

    def update(self, env):
        # reset observation
        self.rgb_observation = self.background_img.copy()

        env_state = env.game_state
        entity_colors = env.ENTITY_COLORS
        entity_ids = env.ENTITY_ID

        # draw passengers
        passengers = env.passenger_locations
        for passenger_loc in passengers:
            passenger_id = env_state[passenger_loc[0]][passenger_loc[1]]
            # if the key in not picked
            if passenger_id in entity_ids['passengers']:
                color = entity_colors[passenger_id]
                self._draw_passengers(passenger_loc, color)

        # draw the agent
        agent_loc = env.agent_loc
        agent_color = entity_colors[entity_ids['agent']]
        self._draw_agent(agent_loc, agent_color)

        # draw the destination
        dest_loc = env.dest_loc
        dest_color = entity_colors[entity_ids['destination']]
        self._draw_dest(dest_loc, dest_color)

    def get_rgb_observation(self):
        return self.rgb_observation

    def render(self):
        global _robot_taxi_main_window, _taxi_robot_canvas
        if _robot_taxi_main_window is None:
            _robot_taxi_main_window = tk.Tk()
            _robot_taxi_main_window.geometry('%dx%d+%d+%d' % (self.screen_width, self.screen_height, 10, 0))
            _robot_taxi_main_window.resizable(False, False)

            _taxi_robot_canvas = tk.Canvas(_robot_taxi_main_window, width=self.screen_width - 1,
                                           height=self.screen_height - 1)
            _taxi_robot_canvas.pack()

        img = ImageTk.PhotoImage(master=_taxi_robot_canvas, image=Image.fromarray(self.rgb_observation))
        _taxi_robot_canvas.create_image((0, 0), image=img, anchor="nw")
        robot_taxi_graphics_sleep(self.frame_time)

