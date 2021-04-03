from config.rl.dqn.efficient_dqn_config import Efficient_DQN_Config
from learning_agents.data_augmentation.augmentations import random_shift_augmentation, Data_Augmentation


class RAD_Config(Efficient_DQN_Config):
    def __init__(self):
        # simply use the default config here
        Efficient_DQN_Config.__init__(self)

    def get_agent_config(self):
        Efficient_DQN_Config.get_agent_config(self)

        self.agent_config.policy_hyper_params.policy_multiple_update = 2
        #################################
        ### Data augmentation config  ###
        #################################
        self.agent_config.policy_hyper_params.aug_class = Data_Augmentation
        self.agent_config.policy_hyper_params.func_augmentations = [random_shift_augmentation]

