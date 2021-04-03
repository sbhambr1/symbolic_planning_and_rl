from abc import ABC, abstractmethod
import argparse

import gym
import numpy as np
import torch


class Agent(ABC):
    """
    Abstract Agent used for all agents.
    Attributes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        log_cfg (ConfigDict): configuration for saving log and checkpoint
        env_name (str) : gym env name for logging
        is_discrete (bool): shows whether the action is discrete
    """

    def __init__(self, env, args):
        """Initialize."""
        self.args = args
        self.env = env

        self.env_name = env.spec.id if env.spec is not None else env.name

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.is_discrete = True
        else:
            self.is_discrete = False

    @abstractmethod
    def select_action(self, state):
        """
        state: np.ndarray
        :return: Union[torch.Tensor, np.ndarray]
        """
        pass

    @abstractmethod
    def update_model(self):
        """
        :return:Tuple[torch.Tensor, ...]
        """
        pass

    @abstractmethod
    def load_params(self, path):
        pass

    @abstractmethod
    def save_params(self, n_step):
        """
        n_step: int
        params: dict
        """
        pass

    @abstractmethod
    def write_log(self, log_value):
        pass

