import glob
import os
import time
from collections import deque

import numpy as np

from learning_agents.rl.value_based_agent.dqn.dqn_agent import DQN_Agent
from learning_agents.rl.value_based_agent.value_based_agent import Value_Based_Agent
from learning_agents.utils.tensor_utils import get_device

device = get_device()


class SDRL_Agent(Value_Based_Agent):
    def __init__(self, env, args, hyper_params, network_cfg, optim_cfg, logger=None):
        """Initialize."""
        Value_Based_Agent.__init__(self, env, args, hyper_params, network_cfg, optim_cfg, logger=logger)

        self.n_subgoal = self.env.n_subgoal
        self.subgoal_success_tracker = [[] for _ in range(self.n_subgoal)]
        self.subgoal_trailing_performance = [0 for _ in range(self.n_subgoal)]
        self.subgoal_avg_score_window = [deque(maxlen=self.args.avg_score_window) for _ in range(self.n_subgoal)]

        self.subgoal_policies = []
        self.n_symbolic_state = self.env.n_symbolic_state
        self.n_symbolic_action = self.env.n_symbolic_action
        self.R_table = np.zeros(shape=(self.n_symbolic_state, self.n_symbolic_action))
        self.ro_table = np.zeros(shape=(self.n_symbolic_state, self.n_symbolic_action))
        self.ro_table_lp = []
        self.current_plan_trace = []
        self.plan_explore = True

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

    # TODO
    def select_action(self, state):
        pass

    def train(self):
        self.env.generate_goal_file(0)
        self.env.cleanup_constraint()
        plan_abandoned = False

        while self.total_step < self.args.max_step and self.i_episode < self.args.max_episode:
            print('[INFO] Starting episode ', self.i_episode)
            self.env.restart()

            self.episode_step = 0
            symbolic_state_action = []
            plan_quality = 0
            score = 0
            done = False
            all_subgoal_learned = True

            self.env.generate_rovalue_from_table(self.ro_table_lp, self.ro_table)
            if self.plan_explore:
                print('[INFO] Explore at plan level: generating new plan ...')
                old_plan = list(self.current_plan_trace)
                self.current_plan_trace = self.env.generate_plan()
                plan_abandoned = False
                if self.current_plan_trace is None:
                    print('[Warning] No plan found at episode ', self.i_episode)
                    self.current_plan_trace = old_plan
                else:
                    print('[INFO] Episode {0}, found plan: {1}'.format(self.i_episode, self.current_plan_trace))

            # Run episode (execute current plan trace)
            subgoal_idx = 0
            goal_not_found = False

            while not self.env.is_terminal() and self.episode_step < self.args.max_traj_len and subgoal_idx < len(
                    self.current_plan_trace) - 1 and not goal_not_found:
                # get current subgoal according to the plan
                sub_goal = self.env.select_subgoal_from_plan(self.current_plan_trace, subgoal_idx)
                symbolic_state, symbolic_action = self.env.get_state_action_from_plan(self.current_plan_trace,
                                                                                      subgoal_idx)
                symbolic_state_action.append((symbolic_state, symbolic_action))
                if (symbolic_state, symbolic_action) not in self.ro_table_lp:
                    self.ro_table_lp.append((symbolic_state, symbolic_action))

                if sub_goal == -1:
                    print('[Warning] subgoal not found in current plan: ', str(self.current_plan_trace))
                    goal_not_found = True
                else:
                    print('[INFO] Episode {0}, subgoal index: {1}, goal: {2}, {3}'.format(self.i_episode, subgoal_idx,
                                                                                          sub_goal,
                                                                                          self.env.goal_meaning[
                                                                                              sub_goal]))
                    # recording training-related information
                    subgoal_losses = list()
                    plan_abandoned = False
                    subgoal_score = 0
                    subgoal_done = False
                    subgoal_agent = self.subgoal_policies[sub_goal]
                    subgoal_agent.episode_step = 0
                    t_begin = time.time()

                    while not self.env.is_terminal() and not subgoal_done and \
                            self.episode_step < self.args.max_traj_len:
                        # interact with the environment
                        state = self.env.get_stacked_state()
                        action = subgoal_agent.select_action(state)
                        next_state, external_reward, _, info = self.env.act(action)

                        subgoal_done, intrinsic_reward = self.env.get_intrinsic_reward(sub_goal)
                        subgoal_score += intrinsic_reward

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

                    # update subgoal agent info
                    subgoal_agent.i_episode += 1
                    subgoal_agent.do_post_episode_update()
                    self.subgoal_avg_score_window[sub_goal].append(subgoal_score)

                    t_end = time.time()
                    avg_time_cost = (t_end - t_begin) / subgoal_agent.episode_step
                    # log general training info
                    if subgoal_losses:
                        avg_loss = np.vstack(subgoal_losses).mean(axis=0)
                        log_value = (avg_loss, subgoal_score, avg_time_cost, np.mean(self.subgoal_avg_score_window[sub_goal]))
                        self.write_subgoal_log(log_value, sub_goal)

            self.i_episode += 1

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

    def write_subgoal_log(self, log_value, sub_goal_idx):
        """Write log about loss and score"""
        loss, score, avg_time_cost, avg_score_window = log_value
        subgoal_log_info = {
            "episode": self.i_episode,
            "score": score,
            "episode step": self.episode_step,
            "total step": self.total_step,
            "epsilon": self.explore_strategy.get_epsilon(self.total_step, self.i_episode),
            "dqn loss": loss[0],
            "avg q values": loss[1],
            "time per each step": avg_time_cost,
            "avg score window": avg_score_window,
        }
        log_info = {}
        for key in subgoal_log_info:
            log_info["sub-goal " + str(sub_goal_idx) + " - " + str(key)] = subgoal_log_info[key]

        print("[INFO] %s\n" % (str(log_info),))
        if self.logger is not None:
            self.logger.log_wandb(log_info, step=self.total_step)

    def write_log(self, log_value):
        """Write log about loss and score"""
        loss, score, avg_time_cost, avg_score_window = log_value
        log_info = {
            "episode": self.i_episode,
            "score": score,
            "episode step": self.episode_step,
            "total step": self.total_step,
            "epsilon": self.explore_strategy.get_epsilon(self.total_step, self.i_episode),
            "dqn loss": loss[0],
            "avg q values": loss[1],
            "time per each step": avg_time_cost,
            "avg score window": avg_score_window,
        }

        print("[INFO] %s\n" % (str(log_info),))
        if self.logger is not None:
            self.logger.log_wandb(log_info, step=self.total_step)

    def update_model(self):
        """Train the model after each episode."""
        pass

    def test(self):
        pass

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain(self):
        """Pretraining steps."""
        pass

    def do_post_episode_update(self, *argv):
        pass
