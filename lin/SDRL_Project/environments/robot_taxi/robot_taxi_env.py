import configparser
import json
import os

import gym
import numpy as np
from addict import Dict

from environments.robot_taxi.robot_taxi_graphics import Robot_Taxi_Graphics


class Robot_Taxi_Env(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    ACTION_ID_TO_DIRECTION = {
        0: (0, 0),  # 'STOP'
        1: (0, 1),  # 'EAST'
        2: (0, -1),  # 'WEST'
        3: (1, 0),  # 'SOUTH'
        4: (-1, 0),  # 'NORTH'
    }

    MAX_PASSENGERS = 10
    RIGHT_PASSENGER_ID = 3
    ENTITY_ID = {
        'agent': 1,
        'destination': 2,
        'passengers': [i + 3 for i in range(0, MAX_PASSENGERS)]
    }

    ENTITY_COLORS = {
        1: (127, 127, 127),  # agent
        2: (0, 0, 0),  # dest
        3: (255, 0, 0),  # the right passenger
        4: (31, 119, 180),
        5: (174, 199, 232),
        6: (255, 127, 14),
        7: (255, 187, 120),
        8: (44, 160, 44),
        9: (152, 223, 138),
        10: (255, 152, 150),
        11: (148, 103, 189),
        12: (197, 176, 213),
        13: (140, 86, 75),
        14: (196, 156, 148),
        15: (227, 119, 194),
    }

    def __init__(self, config_file=None):
        # define the game configuration file (None means default setting)
        self.config_file = config_file
        self.config = self._read_config(config_file)

        # init graphics
        self.display = Robot_Taxi_Graphics(width=self.config['layout_width'],
                                           height=self.config['layout_height'],
                                           zoom=self.config['zoom'],
                                           frame_time=self.config['frame_time'])

        # define the reward range
        self.reward_range = [0, 1]
        # noinspection PyArgumentList
        self.action_space = gym.spaces.Discrete(len(self.ACTION_ID_TO_DIRECTION))
        # define the observation spaces
        # noinspection PyArgumentList
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(self.display.screen_height, self.display.screen_width, 3),
                                                dtype=np.uint8)
        self.name = 'robot_taxi'
        self.spec = Dict({'name': self.name,
                          'action_space': self.action_space,
                          'observation_space': self.observation_space,
                          'reward_range': self.reward_range})
        self.last_action = 0
        self.is_over = False

    # noinspection PyMethodMayBeStatic
    def _read_config(self, config_file):
        config = {}
        if config_file is None:
            config_file = os.path.dirname(os.path.abspath(__file__)) + '/default_config.json'

        with open(config_file) as f:
            config_parser = json.loads(f.read())

        config['layout_width'] = config_parser['layoutWidth']
        config['layout_height'] = config_parser['layoutHeight']

        key_num = config_parser['numPassengers']
        config['passenger_num'] = np.clip(key_num, 1, config['layout_width'])

        zoom = config_parser['zoom']
        config['zoom'] = zoom
        config['frame_time'] = config_parser['frameTime']

        passenger_order = config_parser['passengersOrders']
        config['random_passenger'] = (len(passenger_order) == 0)
        if len(passenger_order) == 0:
            print('[INFO] Robot-Taxi env - randomly generate passenger order')
            passenger_order = np.arange(config['passenger_num'])
            np.random.shuffle(passenger_order)

        config['passenger_order'] = [self.ENTITY_ID['passengers'][i] for i in passenger_order]

        return config

    def _init_game(self):
        self.last_action = 0
        self.is_over = False
        self.game_state = np.zeros(shape=(self.config['layout_height'],
                                          self.config['layout_width']))

        # initial location of the agent: middle of the bottom
        self.agent_loc = [self.config['layout_height'] - 1, int(self.config['layout_width'] / 2)]
        self.game_state[self.agent_loc[0], self.agent_loc[1]] = self.ENTITY_ID['agent']

        # init the door: right-top corner
        self.dest_loc = [0, self.config['layout_width'] - 1]
        self.game_state[self.dest_loc[0], self.dest_loc[1]] = self.ENTITY_ID['destination']

        # init passenger
        self.picked_passenger = None
        self.passenger_locations = [(int(1.0 / 2.0 * self.config['layout_height']),
                                     int(i * float(self.config['layout_width']) / self.MAX_PASSENGERS))
                                    for i in range(1, self.config['passenger_num'] + 1)]
        # random reorder
        if self.config['random_passenger']:
            passenger_order = np.arange(self.config['passenger_num'])
            np.random.shuffle(passenger_order)
            self.config['passenger_order'] = [self.ENTITY_ID['passengers'][i] for i in passenger_order]

        for i, loc in enumerate(self.passenger_locations):
            h, w = loc
            passenger_id = self.config['passenger_order'][i]
            self.game_state[h][w] = passenger_id

    def _move_agent(self, location):
        """ set the location of the agent """
        # clear last location
        self.game_state[self.agent_loc[0], self.agent_loc[1]] = 0

        # check if it picks up any passenger
        for passenger in self.passenger_locations:
            if list(passenger) == location:
                self.picked_passenger = int(self.game_state[location[0], location[1]])

        # set new location
        self.agent_loc = location
        self.game_state[self.agent_loc[0], self.agent_loc[1]] = self.ENTITY_ID['agent']

    def _get_obs(self):
        self.display.update(self)
        return self.display.get_rgb_observation()

    def get_agent_last_move(self):
        return self.last_action

    def _is_pos_valid(self, pos_x, pos_y):
        if pos_x < 0 or pos_x >= self.config['layout_height']:
            return False
        if pos_y < 0 or pos_y >= self.config['layout_width']:
            return False
        return True

    def _update(self, action):
        pos_x = self.agent_loc[0]
        pos_y = self.agent_loc[1]

        movement = self.ACTION_ID_TO_DIRECTION[action]
        new_pos_x = pos_x + movement[0]
        new_pos_y = pos_y + movement[1]
        # if the movement is not valid
        if not self._is_pos_valid(new_pos_x, new_pos_y):
            self.last_action = 0
            return

        # update the map
        self._move_agent([new_pos_x, new_pos_y])

    def step(self, action):
        # noinspection PyTypeChecker
        action = int(action)
        self.last_action = action

        info = {}
        reward = 0
        if not self.is_over:
            self._update(action)

            if self.agent_loc == self.dest_loc:
                self.is_over = True
                if self.picked_passenger is not None and self.picked_passenger == self.RIGHT_PASSENGER_ID:
                    reward = 1
            return self._get_obs(), reward, self.is_over, info
        # game is over
        else:
            return self._get_obs(), reward, self.is_over, info

    def get_last_rgb_obs(self):
        return self._get_obs()

    def reset(self):
        self._init_game()
        return self._get_obs()

    def render(self, mode='human'):
        if mode == 'human':
            self.display.render()
        else:
            return self._get_obs()

    def get_subgoal(self):
        """
        sub-goals: [navigate for pickup, navigate for drop-off]
        """
        if self.picked_passenger is None:
            return 0
        if self.picked_passenger is not None and self.picked_passenger == self.RIGHT_PASSENGER_ID and self.agent_loc != self.dest_loc:
            return 1

    def subgoal_remapping(self):
        if self.get_subgoal() == 0:
            return 'navigate to pickup passenger'
        if self.get_subgoal() == 1:
            return 'navigate to drop-off passenger'
