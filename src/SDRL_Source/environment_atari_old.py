# Hoang M. Le
# California Institute of Technology
# hmle@caltech.edu
# ===================================================================================================================

from hyperparameters_new import *
import sys
import os
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw
import gym
import os

logger = logging.getLogger(__name__)
np.random.seed(SEED)


class ALEEnvironment:
    def __init__(self, rom_file, args):
        self.gym_env = gym.make("MontezumaRevenge-v0")
        self.ale = self.gym_env.ale
        """
        @Lin: states for DRL agent are stacked consecutive frames, e.g. in the shape of 4*64*64
        """
        self.histLen = 4

        self.ale.setInt('frame_skip', args.frame_skip)
        self.ale.setFloat('repeat_action_probability', 0.0)
        self.ale.setBool('color_averaging', args.color_averaging)

        # if args.random_seed:
        #  self.ale.setInt('random_seed', args.random_seed)
        self.ale.setInt('random_seed', 0)  # hoang addition to fix the random seed across all environment

        if args.minimal_action_set:
            self.actions = self.ale.getMinimalActionSet()
            logger.info("Using minimal action set with size %d" % len(self.actions))
        else:
            self.actions = self.ale.getLegalActionSet()
            logger.info("Using full action set with size %d" % len(self.actions))
        logger.debug("Actions: " + str(self.actions))

        self.screen_width = args.screen_width
        self.screen_height = args.screen_height

        self.mode = "train"
        self.life_lost = False
        self.initSrcreen = self.getScreen()
        print("size of screen is:", self.initSrcreen.shape)
        im = Image.fromarray(self.initSrcreen)
        im.save('initial_screen.jpeg')

        """
        @Lin: here they use hard-coded sub-goals
        """
        self.goalSet = []
        # goal 0
        self.goalSet.append([[69, 68], [73, 71]])  # Lower Right Ladder. This is the box for detecting first subgoal
        # self.goalSet.append([[11, 58], [15, 66]]) # lower left ladder 3
        # self.goalSet.append([[11, 68], [15, 71]])  # lower left ladder 3
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
        self.reachedGoal = [0, 0, 0, 0, 0, 0, 0]
        self.histState = self.initializeHistState()

    def initializeHistState(self):
        """
        @Lin: 4 stacked frames as RL state
        """
        histState = np.concatenate((self.getState(), self.getState()), axis=2)
        histState = np.concatenate((histState, self.getState()), axis=2)
        histState = np.concatenate((histState, self.getState()), axis=2)
        return histState

    def numActions(self):
        return len(self.actions)

    def resetGoalReach(self):
        self.reachedGoal = [0, 0, 0, 0, 0, 0, 0, 0]

    def restart(self):
        self.gym_env.reset()
        self.life_lost = False
        self.reachedGoal = [0, 0, 0, 0, 0, 0, 0]
        for i in range(19):
            self.act(0)  # wait for initialization
        self.histState = self.initializeHistState()
        self.agentLastX = self.agentOriginLoc[0]
        self.agentLastY = self.agentOriginLoc[1]

    def beginNextLife(self):
        self.life_lost = False
        self.reachedGoal = [0, 0, 0, 0, 0, 0, 0]
        for i in range(19):
            self.act(0)  # wait for initialization
        self.histState = self.initializeHistState()
        self.agentLastX = self.agentOriginLoc[0]
        self.agentLastY = self.agentOriginLoc[1]

    def act(self, action):
        lives = self.ale.lives()
        observation, reward, done, info = self.gym_env.step(self.actions[action])
        self.life_lost = (not lives == self.ale.lives())
        currState = self.getState()
        self.histState = np.concatenate((self.histState[:, :, 1:], currState), axis=2)
        return reward

    def getScreen(self):
        screen = self.ale.getScreenGrayscale()
        resized = cv2.resize(screen, (self.screen_width, self.screen_height))
        return resized

    def getScreenRGB(self):
        screen = self.ale.getScreenRGB()
        resized = cv2.resize(screen, (self.screen_width, self.screen_height))
        # resized = screen
        return resized

    def getAgentLoc(self, img):
        #  img = self.getScreenRGB()

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
        return (mean_x, mean_y)

    def getDevilLoc(self, img):
        #    img = self.getScreenRGB()
        # man = [0, 16, 2]
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
        return (mean_x, mean_y)

    def distanceReward(self, lastGoal, goal):
        if (lastGoal == -1):
            lastGoalCenter = self.agentOriginLoc
        else:
            lastGoalCenter = self.goalCenterLoc[lastGoal]
        goalCenter = self.goalCenterLoc[goal]
        agentX, agentY = self.getAgentLoc()
        dis = np.sqrt(
            (goalCenter[0] - agentX) * (goalCenter[0] - agentX) + (goalCenter[1] - agentY) * (goalCenter[1] - agentY))
        disLast = np.sqrt((lastGoalCenter[0] - agentX) * (lastGoalCenter[0] - agentX) + (lastGoalCenter[1] - agentY) * (
                    lastGoalCenter[1] - agentY))
        disGoals = np.sqrt((goalCenter[0] - lastGoalCenter[0]) * (goalCenter[0] - lastGoalCenter[0]) + (
                    goalCenter[1] - lastGoalCenter[1]) * (goalCenter[1] - lastGoalCenter[1]))
        return 0.001 * (disLast - dis) / disGoals

    # add color channel for input of network
    def getState(self):
        screen = self.ale.getScreenGrayscale()
        resized = cv2.resize(screen, (self.screen_width, self.screen_height))
        return np.reshape(resized, (84, 84, 1))

    def getStackedState(self):
        return self.histState

    def isTerminal(self):
        if self.mode == 'train':
            return self.ale.game_over() or self.life_lost
        return self.ale.game_over()

    def isGameOver(self):
        return self.ale.game_over()

    def isLifeLost(self):
        return self.life_lost

    def reset(self):
        self.gym_env.reset()
        self.life_lost = False

    def goalReached(self, goal):
        # if goal in [0,2,4,6]: # those are original task where bounding boxes are used to detect the location of agents
        subset = [0, 2, 3, 4,
                  6]  # those are original task where bounding boxes are used to detect the location of agents
        if goal in subset:
            # goal_index = goal/2
            goal_index = subset.index(goal)
            goalPosition = self.goalSet[goal_index]
            goalScreen = self.initSrcreen
            stateScreen = self.getScreen()
            count = 0
            for y in range(goalPosition[0][0], goalPosition[1][0]):
                for x in range(goalPosition[0][1], goalPosition[1][1]):
                    if goalScreen[x][y] != stateScreen[x][y]:
                        count = count + 1
            # 30 is total number of pixels of agent
            if float(count) / 30 > 0.3:
                self.reachedGoal[goal] = 1
                return True
        if goal == 1:
            # detect if agent is to the left of the devil
            #    return self.agent_left_devil()
            return self.detect_left_ladder()
        ############## -- DML modified -- ###########
        # if goal == 4:
        #     # detect if agent is to the right of the devil
        # #    return self.agent_right_devil()
        #     return self.detect_right_ladder()
        ################# -- end -- ###########
        if goal == 5:
            # detect if the agent is back to the original location
            return self.original_location_reached()
        return False

    def detect_right_ladder(self):
        goalPosition = self.goalSet[0]
        goalScreen = self.initSrcreen
        stateScreen = self.getScreen()
        count = 0
        for y in range(goalPosition[0][0], goalPosition[1][0]):
            for x in range(goalPosition[0][1], goalPosition[1][1]):
                if goalScreen[x][y] != stateScreen[x][y]:
                    count = count + 1
        # 30 is total number of pixels of agent
        if float(count) / 30 > 0.3:
            goal = 5
            self.reachedGoal[goal] = 1
            return True
        return False

    def detect_left_ladder(self):
        goalPosition = self.goalSet[2]
        goalScreen = self.initSrcreen
        stateScreen = self.getScreen()
        count = 0
        for y in range(goalPosition[0][0], goalPosition[1][0]):
            for x in range(goalPosition[0][1], goalPosition[1][1]):
                if goalScreen[x][y] != stateScreen[x][y]:
                    count = count + 1
        # 30 is total number of pixels of agent
        if float(count) / 30 > 0.3:
            goal = 5
            self.reachedGoal[goal] = 1
            return True
        return False

    def original_location_reached(self):
        img = self.getScreenRGB()
        (x, y) = self.getAgentLoc(img)
        #  print "Agent's location:",x,y
        if abs(x - 42) <= 2 and abs(y - 33) <= 2:
            return True
        else:
            return False

    def pause(self):
        os.system('read -s -n 1 -p "Press any key to continue...\n"')

    def agent_left_devil(self):
        img = self.ale.getScreenRGB()
        (x, y) = self.getAgentLoc(img)
        (a, b) = self.getDevilLoc(img)
        #  print "Agent's location:",x,y
        #  print "Devil's location:", a,b
        if (a - x > 40) and (abs(y - b) <= 40):
            return True
        else:
            return False

    def agent_right_devil(self):
        img = self.getScreenRGB()
        (x, y) = self.getAgentLoc(img)
        (a, b) = self.getDevilLoc(img)
        # print "Agent's location:",x,y
        # print "Devil's location:",a,b

        # if (x-a > 25) and (abs(y-b) <= 40):
        if (x - a > 40) and (abs(y - b) <= 40):
            return True
        else:
            return False

    def goalNotReachedBefore(self, goal):
        if (self.reachedGoal[goal] == 1):
            return False
        return True
