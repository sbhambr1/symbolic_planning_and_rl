import glob
import os
import time

import numpy as np
import torch
import torch.optim as optim

from learning_agents.architectures.cnn import Conv2d_MLP_Model
from torch.nn.utils import clip_grad_norm_
from learning_agents.rl.value_based_agent.dqn import dqn_utils
from learning_agents.common import common_utils
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
        self.option_learned = [False for _ in range(self.n_subgoal)]

        self.subgoal_policies = []
        self.n_symbolic_state = self.env.n_symbolic_state
        self.n_symbolic_action = self.env.n_symbolic_action
        self.R_table = np.zeros(shape=(self.n_symbolic_state, self.n_symbolic_action))
        self.ro_table = np.zeros(shape=(self.n_symbolic_state, self.n_symbolic_action))
        self.ro_table_lp = []
        self.current_plan_trace = []
        self.plan_explore = False

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

    # TODO
    def update_model(self):
        """Train the model after each episode."""
        pass

    def train(self):
        self.env.generate_goal_file(0)
        self.env.cleanup_constraint()
        plan_abandoned = False

        while self.total_step < self.args.max_step and self.i_episode < self.args.max_episode:
            print('[INFO] Starting episode ', self.i_episode)
            self.env.restart()

            self.episode_step = 0
            plan_quality = 0
            score = 0
            done = False
            t_begin = time.time()

            self.env.generate_rovalue_from_table(self.ro_table_lp, self.ro_table)
            if self.plan_explore:
                print('[INFO] Explore at plan level: generating new plan ...')
                old_plan = list(self.current_plan_trace)
                self.current_plan_trace = self.env.generate_plan()
                plan_abandoned = False
                if self.current_plan_trace is None:
                    print('[Warning] No plan found at episode ', self.i_episode)
                    self.current_plan_trace = old_plan







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

    # TODO
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

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain(self):
        """Pretraining steps."""
        pass

    def do_post_episode_update(self, *argv):
        pass
