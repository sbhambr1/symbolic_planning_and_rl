from config.rl.dqn.efficient_dqn_config import Efficient_DQN_Config
from learning_agents.data_augmentation.augmentations import random_shift_augmentation
from learning_agents.data_augmentation.torch_augmentations import Torch_Data_Augmentation, torch_random_shift


class CURL_Dueling_Config(Efficient_DQN_Config):
    def __init__(self):
        super(CURL_Dueling_Config, self).__init__()

    def get_agent_config(self):
        super(CURL_Dueling_Config, self).get_agent_config()

        ############################
        ## Hyper Parameter Config ##
        ############################
        # default: 1.0 except 0.1 for 'pong', 'boxing', 'private_eye', 'freeway'
        self.agent_config.policy_hyper_params.cpc_loss_weight = 1.0
        self.agent_config.policy_hyper_params.momentum_tau = 1 - 0.999

        #################################
        ### Data augmentation config  ###
        #################################
        self.agent_config.policy_hyper_params.aug_class = Torch_Data_Augmentation
        self.agent_config.policy_hyper_params.func_augmentations = [torch_random_shift]




