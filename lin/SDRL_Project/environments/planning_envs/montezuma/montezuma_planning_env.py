import math
import os
import random

import numpy as np
import cv2

from environments.make_atari_env import make_atari_env
from learning_agents.planning_rl.planners import bc_planner


class Montezuma_Planning_Env:
    def __init__(self, env_name, agent_config):
        self.agent_config = agent_config
        self.env = make_atari_env(env_name=env_name, agent_config=agent_config)
        self.ale = self.env.ale

        self.screen_width = agent_config.sys_args.frame_size
        self.screen_height = agent_config.sys_args.frame_size
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        # symbolic representation
        # 6 locations, doubled with key picked, in total, 8 states, 1 good terminal (-2), 1 bad terminate (-3)
        self.n_symbolic_state = 14
        # move to right ladder, move to key, move to left ladder, move to door, move to left of devil, move to initial
        self.n_symbolic_action = 6

        self.mode = "train"
        self.life_lost = False
        self.init_screen = self.get_screen_gray()

        # files for planning
        self.env_base_dir = os.path.dirname(os.path.abspath(__file__))
        self.constraint_file = os.path.join(self.env_base_dir, 'constraints.lp')
        self.goal_file = os.path.join(self.env_base_dir, 'goal.lp')
        self.domain_file = os.path.join(self.env_base_dir, 'montezuma.lp')
        self.q_file = os.path.join(self.env_base_dir, 'q.lp')
        self.init_file = os.path.join(self.env_base_dir, 'initial.lp')

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
        self.reached_goal = [0, 0, 0, 0, 0, 0, 0]

        self.stacked_state = self._get_init_stacked_state()

    def _get_init_stacked_state(self):
        """
        @Lin: 4 stacked frames as RL state
        """
        return self.env.reset()

    def num_actions(self):
        return self.env.action_space.shape[0]

    def reset_goal_reach(self):
        self.reached_goal = [0, 0, 0, 0, 0, 0, 0, 0]

    def restart(self):
        self.env.reset()
        self.life_lost = False
        self.reached_goal = [0, 0, 0, 0, 0, 0, 0]
        for i in range(19):
            self.act(0)  # wait for initialization
        self.stacked_state = self._get_init_stacked_state()
        self.agentLastX = self.agentOriginLoc[0]
        self.agentLastY = self.agentOriginLoc[1]

    def begin_next_life(self):
        self.life_lost = False
        self.reached_goal = [0, 0, 0, 0, 0, 0, 0]
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
        return reward

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

    def distance_reward(self, last_goal, goal):
        if (last_goal == -1):
            lastGoalCenter = self.agentOriginLoc
        else:
            lastGoalCenter = self.goalCenterLoc[last_goal]
        goalCenter = self.goalCenterLoc[goal]
        agentX, agentY = self.get_agent_loc()
        dis = np.sqrt(
            (goalCenter[0] - agentX) * (goalCenter[0] - agentX) + (goalCenter[1] - agentY) * (goalCenter[1] - agentY))
        disLast = np.sqrt((lastGoalCenter[0] - agentX) * (lastGoalCenter[0] - agentX) + (lastGoalCenter[1] - agentY) * (
                lastGoalCenter[1] - agentY))
        disGoals = np.sqrt((goalCenter[0] - lastGoalCenter[0]) * (goalCenter[0] - lastGoalCenter[0]) + (
                goalCenter[1] - lastGoalCenter[1]) * (goalCenter[1] - lastGoalCenter[1]))
        return 0.001 * (disLast - dis) / disGoals

    def get_stacked_state(self):
        return self.stacked_state

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

    def generate_plan(self):
        return bc_planner.compute_plan(clingo_path='clingo', initial=self.init_file, goal=self.goal_file,
                                       planning=self.domain_file, q_value=self.q_file, constraint=self.constraint_file,
                                       log_dir=self.env_base_dir, verbose=True)

    def generate_rovalue_from_table(self, ro_table_lp, ro_table):
        """
        Output new q value files for more accurate planning
        """
        q_file = open(self.q_file, "w")
        for (state, action) in ro_table_lp:
            logical_state = self.state_remapping(state)
            logical_action = self.action_remapping(action)
            q_rule = "ro(" + logical_state + "," + logical_action + "," + str(
                int(math.floor(ro_table[state, action]))) + ").\n"
            q_file.write(q_rule)
        q_file.close()

    def generate_goal_file(self, plan_quality):
        """
        Output new goal file for planning
        """
        goal_file = open(self.goal_file, "w")
        goal_file.write("#program check(k).\n")
        goal_file.write(":- query(k), cost(C,k), C <= " + str(plan_quality) + ".\n")
        goal_file.write(":- query(k), cost(0,k).")
        goal_file.close()

    def update_constraint(self, state_id, action_id):
        state = self.state_remapping(state_id, include_timestamp=True)
        action = self.action_remapping(action_id, include_timestamp=True)
        constraint = ":-" + state + "," + action + ".\n"
        f = open(self.constraint_file, "a")
        f.write("#program step(k).\n")
        f.write(constraint)
        f.close()

    def cleanup_constraint(self):
        open(self.constraint_file, 'w').close()

    @staticmethod
    def select_subgoal_from_plan(plan_trace, i):
        current_unit = plan_trace[i]
        current_fluent = current_unit[2]
        next_unit = plan_trace[i + 1]
        next_fluent = next_unit[2]

        # Make sure the goal number here maps correctly to bounding boxes in environment_atari.py
        if ("at(plat1)" in current_fluent) and ("at(lower_right_ladder)" in next_fluent) and (
                "picked(key)" not in next_fluent):
            return 0
        if ("at(lower_right_ladder)" in current_fluent) and ("at(devilleft)" in next_fluent):
            return 1
        if ("at(devilleft)" in current_fluent) and ("at(key)" in next_fluent):
            return 2
        if ("at(key)" in current_fluent) and ("at(lower_left_ladder)" in next_fluent):
            return 3
        if ("at(lower_left_ladder)" in current_fluent) and ("at(lower_right_ladder)" in next_fluent):
            return 4
        if ("at(lower_right_ladder)" in current_fluent) and ("at(plat1)" in next_fluent):
            return 5
        if ("at(plat1)" in current_fluent) and ("at(right_door)" in next_fluent):
            return 6
        return -1

    def get_state_action_from_plan(self, plan_trace, i):
        unit = plan_trace[i]
        action = unit[1]
        fluent = unit[2]
        return self.state_mapping(fluent), self.action_mapping(action)

    @staticmethod
    def action_mapping(action):
        if 'move(lower_right_ladder)' in action:
            return 0
        if 'move(lower_left_ladder)' in action:
            return 1
        if 'move(key)' in action:
            return 2
        if 'move(right_door)' in action:
            return 3
        if 'move(devilleft)' in action:
            return 4
        if 'move(plat1)' in action:
            return 5

    @staticmethod
    def state_mapping(fluent):
        """
        Symbolic state to goal mapping
        """
        if ("at(lower_right_ladder)" in fluent) and ("picked(key)" not in fluent):
            return 0
        if ("at(key)" in fluent) and ("picked(key)" in fluent):
            return 1
        if ("at(lower_right_ladder)" in fluent) and ("picked(key)" in fluent):
            return 2
        if ("at(right_door)" in fluent) and ("picked(key)" in fluent):
            return 3
        if ("at(right_door)" in fluent) and ("picked(key)" not in fluent):
            return 4
        if ("at(devilleft)" in fluent):
            return 5
        if ("at(plat1)" in fluent) and ("picked(key)" in fluent):
            return 6
        if ("at(lower_left_ladder)" in fluent) and ("picked(key)" in fluent):
            return 7
        if ("at(lower_left_ladder)" in fluent) and ("picked(key)" not in fluent):
            return 8
        return -1

    @staticmethod
    def action_remapping(action_id, include_timestamp=False):
        timestamp_str = ',k)' if include_timestamp else ')'
        if action_id == 0:
            return 'move(lower_right_ladder' + timestamp_str
        if action_id == 1:
            return 'move(lower_left_ladder' + timestamp_str
        if action_id == 2:
            return 'move(key' + timestamp_str
        if action_id == 3:
            return 'move(right_door' + timestamp_str
        if action_id == 4:
            return 'move(devilleft' + timestamp_str
        if action_id == 5:
            return 'move(plat1' + timestamp_str
        return ''

    @staticmethod
    def state_remapping(fluent_id, include_timestamp=False):
        timestamp_str = ',k)' if include_timestamp else ')'
        if fluent_id == -1:
            return 'at(plat1' + timestamp_str
        if fluent_id == 0:
            return 'at(lower_right_ladder' + timestamp_str
        elif fluent_id == 1:
            return '(at(key),picked(key)' + timestamp_str
        elif fluent_id == 2:
            return '(at(lower_right_ladder),picked(key)' + timestamp_str
        elif fluent_id == 3:
            return '(at(right_door),picked(key)' + timestamp_str
        elif fluent_id == 4:
            return 'at(right_door' + timestamp_str
        elif fluent_id == 5:
            return 'at(devilleft' + timestamp_str
        elif fluent_id == 6:
            return '(at(plat1),picked(key)' + timestamp_str
        elif fluent_id == 7:
            return '(at(lower_left_ladder),picked(key)' + timestamp_str
        elif fluent_id == 8:
            return 'at(lower_left_ladder' + timestamp_str
        return ''

    @staticmethod
    def obtain_key(previous_state, next_state):
        """
        Return true if current sub-goal is to pick up the key
        """
        if ("picked(key)" not in previous_state) and ("picked(key)" in next_state):
            return True
        else:
            return False

    @staticmethod
    def open_door(previous_state, next_state):
        """
        Return true if current sub-goal is to open the door
        To open the door, the agent must have picked the key and go to the right door
        """
        if ("picked(key)" in previous_state) and ("at(right_door)" not in previous_state) and (
                "picked(key)" in next_state) and (
                "at(right_door)" in next_state):
            return True
        else:
            return False

    @staticmethod
    def calculate_plan_quality(ro_table, state_action):
        plan_quality = 0
        for (state, action) in state_action:
            plan_quality += int(math.floor(ro_table[state, action]))
        return plan_quality

    @staticmethod
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    @staticmethod
    def throw_dice(threshold):
        rand = random.uniform(0, 1)
        if rand < threshold:
            return True
        else:
            return False
