import argparse

from config.human.feedback.tamer.dqn_tamer_config import DQN_TAMER_Config
from config.human_study.default_human_config import Default_Human_Config
from config.rl.dqn.default_dqn_config import DQN_Config
from learning_agents.data_augmentation.augmentations import Data_Augmentation
from learning_agents.data_augmentation.expand.expand_aug_functions import expand_gaussian_blur


class EXPAND_TAMER_Config(DQN_TAMER_Config):
    def __init__(self):
        DQN_TAMER_Config.__init__(self)

    def get_agent_config(self):
        DQN_TAMER_Config.get_agent_config(self)

        ############################
        ## Hyper Parameter Config ##
        ############################
        self.agent_config.policy_hyper_params.batch_size = 64
        self.agent_config.policy_hyper_params.lambda_value_invariant = 0
        self.agent_config.policy_hyper_params.lambda_policy_invariant = 1.0
        self.agent_config.policy_hyper_params.human_feedback_weight = 1.0
        self.agent_config.policy_hyper_params.gradient_clip = None

        # Epsilon Greedy
        self.agent_config.policy_hyper_params.explore_strategy = 'epsilon-greedy'
        self.agent_config.policy_hyper_params.epsilon_strategy = 'exponential-episode'
        self.agent_config.policy_hyper_params.max_epsilon = 1.0
        self.agent_config.policy_hyper_params.min_epsilon = 0.01  # open-ai baselines: 0.01
        self.agent_config.policy_hyper_params.epsilon_decay = 0.9  # default: 0.9995

        self.agent_config.policy_hyper_params.alpha_human = 1.0
        self.agent_config.policy_hyper_params.human_feedback_multiple_update = 1
        self.agent_config.policy_hyper_params.feedback_batch_size = 64
        self.agent_config.policy_hyper_params.feedback_buffer_size = int(5e4)

        self.agent_config.policy_optim_cfg.weight_decay = 1e-8

        self.agent_config.policy_hyper_params.aug_class = Data_Augmentation
        self.agent_config.policy_hyper_params.expand_augmentations = [expand_gaussian_blur]



