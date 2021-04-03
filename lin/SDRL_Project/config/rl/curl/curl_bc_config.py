from config.lfd.bc.default_bc_config import Default_BC_Config
from learning_agents.data_augmentation.torch_augmentations import Torch_Data_Augmentation, torch_no_augmentation, \
    torch_random_shift


class CURL_BC_Config(Default_BC_Config):
    def __init__(self):
        # simply use the default config here
        super(CURL_BC_Config, self).__init__()

    def get_agent_config(self):
        super(CURL_BC_Config, self).get_agent_config()

        self.agent_config.policy_hyper_params.momentum_tau = 1 - 0.999
        self.agent_config.policy_hyper_params.contrastive_epoch = None

        #################################
        ### Data augmentation config  ###
        #################################
        self.agent_config.policy_hyper_params.aug_class = Torch_Data_Augmentation
        self.agent_config.policy_hyper_params.func_augmentations = [torch_random_shift]



