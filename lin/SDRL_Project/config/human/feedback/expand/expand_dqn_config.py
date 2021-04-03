import argparse

from config.human_study.default_human_config import Default_Human_Config
from config.rl.dqn.default_dqn_config import DQN_Config
from learning_agents.data_augmentation.augmentations import Data_Augmentation
from learning_agents.data_augmentation.expand.expand_aug_functions import expand_gaussian_blur


class EXPAND_DQN_Config(DQN_Config, Default_Human_Config):
    def __init__(self):
        DQN_Config.__init__(self)
        Default_Human_Config.__init__(self)

    def parse_sys_args(self):
        """
        Read command line arguments, save into agent config
        :return: ArgumentParser
        """
        super(EXPAND_DQN_Config, self).parse_sys_args()
        parser = argparse.ArgumentParser(description="EXPAND related arguments.")

        parser.add_argument("--no-policy-invariant", action="store_true", default=False,
                            help="whether to turn off policy invariant loss")
        parser.add_argument("--no-value-invariant", action="store_true", default=False,
                            help="whether to turn off value invariant loss")
        parser.add_argument("--no-avg-advantage-loss", action="store_true", default=False,
                            help="whether to use avg advantage loss")

        # save system args in agent config
        args, unknown = parser.parse_known_args()
        for arg in vars(args):
            self.agent_config.sys_args[arg] = getattr(args, arg)

        return parser

    def get_agent_config(self):
        DQN_Config.get_agent_config(self)
        Default_Human_Config.get_agent_config(self)

        ############################
        ## Hyper Parameter Config ##
        ############################
        self.agent_config.policy_hyper_params.batch_size = 64
        self.agent_config.policy_hyper_params.advantage_margin = 0.05
        self.agent_config.policy_hyper_params.lambda_advantage = 1.0
        self.agent_config.policy_hyper_params.human_feedback_weight = 1.0

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



