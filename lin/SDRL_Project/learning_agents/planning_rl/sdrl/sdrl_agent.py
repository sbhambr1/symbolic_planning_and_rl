import glob
import os
import time
from collections import deque

import numpy as np

from learning_agents.rl.value_based_agent.dqn.dqn_agent import DQN_Agent
from learning_agents.rl.value_based_agent.value_based_agent import Value_Based_Agent
from learning_agents.utils.tensor_utils import get_device

device = get_device()
DEBUG_INFO = False


class SDRL_Agent(Value_Based_Agent):
    def __init__(self, env, args, hyper_params, network_cfg, optim_cfg, logger=None):
        """Initialize."""
        Value_Based_Agent.__init__(self, env, args, hyper_params, network_cfg, optim_cfg, logger=logger)

        self.n_subgoal = self.env.n_subgoal
        self.subgoal_success_tracker = [[] for _ in range(self.n_subgoal)]
        self.subgoal_avg_score_window = [deque(maxlen=self.args.avg_score_window) for _ in range(self.n_subgoal)]

        self.subgoal_policies = []
        self._init_network()

    # pylint: disable=attribute-defined-outside-init
    def _init_network(self):
        """Initialize networks and optimizers."""
        for _ in range(self.n_subgoal):
            self.subgoal_policies.append(
                DQN_Agent(self.env, self.args, self.hyper_params, self.network_cfg, self.optim_cfg, logger=self.logger))
        print('[INFO] Initialize {0} DQN agents as the sub-goal policies'.format(self.n_subgoal))

    def _init_from_file(self):
        # load the optimizer and model parameters
        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

    def train(self):
        self.subgoal_avg_score_window = [deque(maxlen=self.args.avg_score_window) for _ in range(self.n_subgoal)]

        while self.total_step < self.args.max_step and self.i_episode < self.args.max_episode:
            if DEBUG_INFO:
                print('[INFO] Starting episode ', self.i_episode)
            self.env.restart()

            self.episode_step = 0
            score = 0

            while not self.env.is_terminal() and self.episode_step < self.args.max_traj_len:
                sub_goal = self.env.get_current_subgoal()
                if DEBUG_INFO:
                    print('[INFO] Episode: {0}, episode step: {1}, total step: {4}, current subgoal {2}: {3}.'.format(
                        self.i_episode, self.episode_step, sub_goal,
                        self.env.goal_meaning[sub_goal], self.total_step))

                # recording training-related information
                subgoal_losses = list()
                subgoal_score = 0
                subgoal_done = False
                # get the sub-policy
                subgoal_agent = self.subgoal_policies[sub_goal]
                subgoal_agent.episode_step = 0
                t_begin = time.time()

                while not self.env.is_terminal() and not subgoal_done and \
                        subgoal_agent.episode_step < self.hyper_params.max_goal_step:
                    # interact with the environment
                    state = self.env.get_stacked_state()
                    action = subgoal_agent.select_action(state)
                    next_state, external_reward, _, info = self.env.act(action)
                    score += external_reward

                    intrinsic_reward, subgoal_done = self.env.get_intrinsic_reward(sub_goal)
                    subgoal_score += intrinsic_reward
                    if subgoal_agent.episode_step + 1 >= self.hyper_params.max_goal_step:
                        intrinsic_reward -= 1

                    # save the new transition
                    transition = (state, action, intrinsic_reward, next_state, subgoal_done, info)
                    subgoal_agent.add_transition_to_memory(transition)

                    # update subgoal agent info
                    subgoal_agent.episode_step += 1
                    subgoal_agent.total_step += 1
                    # update SDRL agent info
                    self.episode_step += 1
                    self.total_step += 1

                    if subgoal_agent.total_step % self.args.save_period == 0:
                        subgoal_agent.save_params(n_step=subgoal_agent.total_step,
                                                  prefix='sub_policy_{0}'.format(sub_goal))

                    # train the agent
                    if len(subgoal_agent.memory) >= subgoal_agent.hyper_params.update_starts_from:
                        if subgoal_agent.total_step % subgoal_agent.hyper_params.train_freq == 0:
                            for _ in range(subgoal_agent.hyper_params.policy_multiple_update):
                                loss = subgoal_agent.update_model()
                                subgoal_losses.append(loss)  # for logging

                self.i_episode += 1
                # update subgoal agent info
                subgoal_agent.i_episode += 1
                subgoal_agent.do_post_episode_update()
                # noinspection PyTypeChecker
                self.subgoal_avg_score_window[sub_goal].append(subgoal_score)

                t_end = time.time()
                avg_time_cost = (t_end - t_begin) / subgoal_agent.episode_step
                # log general training info
                if subgoal_losses:
                    avg_loss = np.vstack(subgoal_losses).mean(axis=0)
                    log_value = (
                        avg_loss, subgoal_score, avg_time_cost, np.mean(self.subgoal_avg_score_window[sub_goal]))
                    self.write_subgoal_log(log_value, sub_goal, subgoal_agent)
                log_value = (sub_goal, score, avg_time_cost)
                self.write_log(log_value)

    def load_params(self, path):
        """Load model and optimizer parameters."""
        # fetch all sub-policies file
        fnames = glob.glob(os.path.join(path, 'policy_subgoal*'))
        for i in range(self.n_subgoal):
            for fname in fnames:
                if 'policy_subgoal_{0}'.format(i) in fname:
                    self.subgoal_policies[i].load_params(fname)
                    break

    def save_params(self, n_step):
        """Save model and optimizer parameters."""
        for i in range(self.n_subgoal):
            self.subgoal_policies[i].save_params(n_step, prefix='policy_subgoal_{0}'.format(i))

    def write_subgoal_log(self, log_value, sub_goal, subgoal_agent):
        """Write log about loss and score"""
        loss, score, avg_time_cost, avg_score_window = log_value
        subgoal_log_info = {
            "episode": subgoal_agent.i_episode,
            "score": score,
            "episode step": subgoal_agent.episode_step,
            "total step": subgoal_agent.total_step,
            "epsilon": subgoal_agent.explore_strategy.get_epsilon(subgoal_agent.total_step, subgoal_agent.i_episode),
            "dqn loss": loss[0],
            "avg q values": loss[1],
            "time per each step": avg_time_cost,
            "avg score window": avg_score_window,
        }
        log_info = {}
        for key in subgoal_log_info:
            log_info["subgoal-" + str(sub_goal) + "-" + str(key)] = subgoal_log_info[key]

        print("[INFO] %s\n" % (str(log_info),))
        if self.logger is not None:
            self.logger.log_wandb(log_info, step=self.total_step)

    def write_log(self, log_value):
        """Write log about loss and score"""
        subgoal, score, avg_time_cost = log_value
        log_info = {
            "episode": self.i_episode,
            "score": score,
            "subgoal": subgoal,
            "total step": self.total_step,
            "episode step": self.episode_step,
        }

        print("[INFO] %s\n" % (str(log_info),))
        if self.logger is not None:
            self.logger.log_wandb(log_info, step=self.total_step)

    def update_model(self):
        """Train the model after each episode."""
        pass

    def select_action(self, state):
        pass

    def test(self):
        pass

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain(self):
        """Pretraining steps."""
        pass

    def do_post_episode_update(self, *argv):
        pass
