from abc import ABC

import gym
import torch

from learning_agents.agent import Agent
from learning_agents.common.prioritized_replay_buffer import PrioritizedReplayBuffer
from learning_agents.common.replay_buffer import ReplayBuffer
from learning_agents.exploration.exploration_strategy import get_strategy
from learning_agents.utils.tensor_utils import get_device

device = get_device()


class Value_Based_Agent(Agent, ABC):
    def __init__(self, env, args, hyper_params, network_cfg, optim_cfg, logger=None):
        Agent.__init__(self, env, args)

        self.episode_step = 0
        self.total_step = 0
        self.i_episode = 0
        self.testing = self.args.test

        self.logger = logger
        self.hyper_params = hyper_params
        self.network_cfg = network_cfg
        self.optim_cfg = optim_cfg

        # get state space info
        self.state_dim = self.env.observation_space.shape[0]
        print('[INFO] Value Based Agent - state shape: ', self.env.observation_space.shape)
        # check if it's single channel or multi channel
        self.state_channel = self.args.frame_stack
        print('[INFO] Value Based Agent - # stack states: ', self.state_channel)
        # get action space info
        self.is_discrete = False
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.is_discrete = True
        else:
            self.action_dim = self.env.action_space.shape[0]
        print('[INFO] Value Based Agent - action dim: ', self.action_dim, ', is discrete: ', self.is_discrete)

        self.use_n_step = hyper_params.n_step > 1
        self.use_prioritized = hyper_params.use_prioritized
        self.per_beta = hyper_params.per_beta if self.use_prioritized else None

        self.explore_strategy = get_strategy(hyper_params.explore_strategy, hyper_params)
        self.epsilon = 0
        if hyper_params.use_noisy_net:
            self.max_epsilon = 0.0
            self.min_epsilon = 0.0
            self.epsilon = 0.0

        self._initialize_buffer()

    # pylint: disable=attribute-defined-outside-init
    def _initialize_buffer(self):
        """Initialize replay buffers."""
        if not self.args.test:
            # replay memory for a single step
            if self.use_prioritized:
                self.memory = PrioritizedReplayBuffer(
                    self.hyper_params.buffer_size,
                    self.hyper_params.batch_size,
                    alpha=self.hyper_params.per_alpha,
                )
            # use ordinary replay buffer
            else:
                self.memory = ReplayBuffer(
                    self.hyper_params.buffer_size,
                    batch_size=self.hyper_params.batch_size,
                    gamma=self.hyper_params.gamma,
                )

            # replay memory for multi-steps
            if self.use_n_step:
                self.memory_n = ReplayBuffer(
                    self.hyper_params.buffer_size,
                    batch_size=self.hyper_params.batch_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                )

    # pylint: disable=no-self-use
    # noinspection PyMethodMayBeStatic
    def _preprocess_state(self, state):
        """Preprocess state so that actor selects an action."""
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(device)
        # if state is a single state, we unsqueeze it
        if len(state.size()) == 3:
            state = state.unsqueeze(0)
        return state

    def add_transition_to_memory(self, transition):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(transition)


