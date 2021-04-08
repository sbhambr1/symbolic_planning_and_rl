import math
import os
import random

import numpy as np
import cv2


class Montezuma_Planning_Env:
    def __init__(self, env_name, agent_config):
        self.agent_config = agent_config
        from environments.make_env import make_env
        self.env = make_env(env_name='MontezumaRevenge', agent_config=agent_config)
        self.spec = self.env.spec
        self.name = self.spec.id if self.spec is not None else 'MontezumaRevenge'
        self.ale = self.env.ale

        self.screen_width = agent_config.sys_args.frame_size
        self.screen_height = agent_config.sys_args.frame_size
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.mode = "train"
        self.life_lost = False
        self.init_screen = self.get_screen_gray()

        """
        Sub-goal definitions
        """
        self.goal_meaning = ['lower right ladder', 'jump to the left of devil', 'key', 'lower left ladder',
                             'lower right ladder', 'central high platform', 'right door']
        self.n_subgoal = len(self.goal_meaning)
        self.goalSet = []
        # goal 0
        self.goalSet.append([[69, 68], [73, 71]])  # Lower Right Ladder. This is the box for detecting first subgoal
        # goal 2
        self.goalSet.append([[7, 41], [11, 45]])  # Key. This will be second sub goal
        # goal 3
        self.goalSet.append([[11, 68], [15, 71]])  # lower left ladder 3
        # goal 4
        self.goalSet.append([[69, 68], [73, 71]])  # Lower Right Ladder again, this will be the third subgoal
        # goal 6
        self.goalSet.append([[70, 20], [73, 35]])  # Right Door. This will be the 4th subgoal
        self.goalCenterLoc = []

        for goal in self.goalSet:
            goalCenter = [float(goal[0][0] + goal[1][0]) / 2, float(goal[0][1] + goal[1][1]) / 2]
            self.goalCenterLoc.append(goalCenter)

        self.agentOriginLoc = [42, 33]
        self.agentLastX = 42
        self.agentLastY = 33
        self.devilLastX = 0
        self.devilLastY = 0
        self.reached_goal = [0 for _ in range(self.n_subgoal)]

        self.stacked_state = self._get_init_stacked_state()

    def _get_init_stacked_state(self):
        """
        @Lin: 4 stacked frames as RL state
        """
        return self.env.reset()

    def get_current_subgoal(self):
        last_unsolved = self.n_subgoal - 1
        for i in [self.n_subgoal - 1 - j for j in range(self.n_subgoal)]:
            if self.reached_goal[i] == 0:
                last_unsolved = i
            else:
                break
        return last_unsolved

    def reset_goal_reach(self):
        self.reached_goal = [0 for _ in range(self.n_subgoal)]

    def restart(self):
        self.env.reset()
        self.life_lost = False
        self.reset_goal_reach()
        for i in range(19):
            self.act(0)  # wait for initialization
        self.stacked_state = self._get_init_stacked_state()
        self.agentLastX = self.agentOriginLoc[0]
        self.agentLastY = self.agentOriginLoc[1]

    def begin_next_life(self):
        self.life_lost = False
        self.reset_goal_reach()
        for i in range(19):
            self.act(0)  # wait for initialization
        self.stacked_state = self._get_init_stacked_state()
        self.agentLastX = self.agentOriginLoc[0]
        self.agentLastY = self.agentOriginLoc[1]

    def act(self, action):
        lives = self.ale.lives()
        next_state, reward, done, info = self.env.step(action)
        self.life_lost = (not lives == self.ale.lives())
        self.stacked_state = next_state
        return next_state, reward, done, info

    def get_screen_gray(self):
        screen = self.ale.getScreenGrayscale()
        resized = cv2.resize(screen, (self.screen_width, self.screen_height))
        return resized

    def get_screen_rgb(self):
        screen = self.ale.getScreenRGB()
        resized = cv2.resize(screen, (self.screen_width, self.screen_height))
        return resized

    def get_agent_loc(self, img=None):
        if img is None:
            img = self.get_screen_rgb()

        man = [200, 72, 72]
        mask = np.zeros(np.shape(img))
        mask[:, :, 0] = man[0]
        mask[:, :, 1] = man[1]
        mask[:, :, 2] = man[2]

        diff = img - mask
        indxs = np.where(diff == 0)
        diff[np.where(diff < 0)] = 0
        diff[np.where(diff > 0)] = 0
        diff[indxs] = 255
        if (np.shape(indxs[0])[0] == 0):
            mean_x = self.agentLastX
            mean_y = self.agentLastY
        else:
            mean_y = np.sum(indxs[0]) / np.shape(indxs[0])[0]
            mean_x = np.sum(indxs[1]) / np.shape(indxs[1])[0]
        self.agentLastX = mean_x
        self.agentLastY = mean_y
        return mean_x, mean_y

    def get_devil_loc(self, img=None):
        if img is None:
            img = self.get_screen_rgb()

        devilColor = [236, 236, 236]
        mask = np.zeros(np.shape(img))
        mask[:, :, 0] = devilColor[0]
        mask[:, :, 1] = devilColor[1]
        mask[:, :, 2] = devilColor[2]
        diff = img - mask
        indxs = np.where(diff == 0)
        diff[np.where(diff < 0)] = 0
        diff[np.where(diff > 0)] = 0
        diff[indxs] = 255
        if (np.shape(indxs[0])[0] == 0):
            mean_x = self.devilLastX
            mean_y = self.devilLastY
        else:
            mean_y = np.sum(indxs[0]) / np.shape(indxs[0])[0]
            mean_x = np.sum(indxs[1]) / np.shape(indxs[1])[0]
        self.devilLastX = mean_x
        self.devilLastY = mean_y
        return mean_x, mean_y

    def get_intrinsic_reward(self, goal):
        reward = 0
        sub_goal_done = self.goal_reached(goal)
        if sub_goal_done:
            reward = 1
        if self.life_lost:
            reward = -1
        return reward, sub_goal_done

    def get_stacked_state(self):
        return np.copy(self.stacked_state)

    def is_terminal(self):
        if self.mode == 'train':
            return self.ale.game_over() or self.life_lost
        return self.ale.game_over()

    def is_game_over(self):
        return self.ale.game_over()

    def is_life_lost(self):
        return self.life_lost

    def reset(self):
        self.env.reset()
        self.life_lost = False

    def goal_reached(self, goal):
        subset = [0, 2, 3, 4,
                  6]  # those are original task where bounding boxes are used to detect the location of agents
        if goal in subset:
            # goal_index = goal/2
            goal_index = subset.index(goal)
            goalPosition = self.goalSet[goal_index]
            goalScreen = self.init_screen
            stateScreen = self.get_screen_gray()
            count = 0
            for y in range(goalPosition[0][0], goalPosition[1][0]):
                for x in range(goalPosition[0][1], goalPosition[1][1]):
                    if goalScreen[x][y] != stateScreen[x][y]:
                        count = count + 1
            # 30 is total number of pixels of agent
            if float(count) / 30 > 0.3:
                self.reached_goal[goal] = 1
                return True
        if goal == 1:
            return self.detect_left_ladder()
        if goal == 5:
            # detect if the agent is back to the original location
            return self.original_location_reached()
        return False

    def detect_right_ladder(self):
        goalPosition = self.goalSet[0]
        goalScreen = self.init_screen
        stateScreen = self.get_screen_gray()
        count = 0
        for y in range(goalPosition[0][0], goalPosition[1][0]):
            for x in range(goalPosition[0][1], goalPosition[1][1]):
                if goalScreen[x][y] != stateScreen[x][y]:
                    count = count + 1
        # 30 is total number of pixels of agent
        if float(count) / 30 > 0.3:
            goal = 5
            self.reached_goal[goal] = 1
            return True
        return False

    def detect_left_ladder(self):
        goalPosition = self.goalSet[2]
        goalScreen = self.init_screen
        stateScreen = self.get_screen_gray()
        count = 0
        for y in range(goalPosition[0][0], goalPosition[1][0]):
            for x in range(goalPosition[0][1], goalPosition[1][1]):
                if goalScreen[x][y] != stateScreen[x][y]:
                    count = count + 1
        # 30 is total number of pixels of agent
        if float(count) / 30 > 0.3:
            goal = 5
            self.reached_goal[goal] = 1
            return True
        return False

    def original_location_reached(self):
        img = self.get_screen_rgb()
        (x, y) = self.get_agent_loc(img)
        #  print "Agent's location:",x,y
        if abs(x - 42) <= 2 and abs(y - 33) <= 2:
            return True
        else:
            return False

    def goal_not_reached_before(self, goal):
        if self.reached_goal[goal] == 1:
            return False
        return True

    @staticmethod
    def throw_dice(threshold):
        rand = random.uniform(0, 1)
        if rand < threshold:
            return True
        else:
            return False
