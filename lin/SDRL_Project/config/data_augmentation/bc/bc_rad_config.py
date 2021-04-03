from config.data_augmentation.bc.default_bc_aug_config import Default_BC_Aug_Config
from learning_agents.data_augmentation.augmentations import random_shift_augmentation, random_cutout_augmentation, \
    Data_Augmentation


class BC_RAD_Config(Default_BC_Aug_Config):
    def __init__(self):
        # simply use the default config here
        Default_BC_Aug_Config.__init__(self)

    def get_agent_config(self):
        Default_BC_Aug_Config.get_agent_config(self)

        #################################
        ### Data augmentation config  ###
        #################################
        self.agent_config.policy_hyper_params.aug_class = Data_Augmentation
        self.agent_config.policy_hyper_params.func_augmentations = [random_shift_augmentation]
        # self.agent_config.policy_hyper_params.func_augmentations = [random_cutout_augmentation]

