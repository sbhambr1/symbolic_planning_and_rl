from config.rl.dqn.default_dqn_config import DQN_Config
from learning_agents.data_augmentation.torch_augmentations import Torch_Data_Augmentation, torch_random_shift


class FERM_DQN_Config(DQN_Config):
    def __init__(self):
        # simply use the default config here
        DQN_Config.__init__(self)

    def get_agent_config(self):
        DQN_Config.get_agent_config(self)

        ############################
        ## Hyper Parameter Config ##
        ############################
        self.agent_config.policy_hyper_params.cpc_loss_weight = 0
        self.agent_config.policy_hyper_params.cpc_pretrain_iter = 1000
        self.agent_config.policy_hyper_params.momentum_tau = 1 - 0.999

        # open-ai baselines default: 0.6, alpha -> 1, full prioritization
        self.agent_config.policy_hyper_params.per_alpha = 0.4
        # beta can start small (for stability concern and anneals towards 1)
        self.agent_config.policy_hyper_params.per_beta = 0.6
        self.agent_config.policy_hyper_params.per_eps = 1e-3
        self.agent_config.policy_hyper_params.per_eps_demo = 1.0,  # default: 1.0

        # dqfd pretrain config
        self.agent_config.policy_hyper_params.dqfd_pretrain_iter = 1000
        self.agent_config.policy_hyper_params.dqfd_detach_encoder = False
        self.agent_config.policy_hyper_params.dqfd_margin = 0.8
        self.agent_config.policy_hyper_params.dqfd_tau = 5e-3

        self.agent_config.policy_hyper_params.train_test_split = 1.0
        self.agent_config.policy_hyper_params.n_step = 5

        ########################
        ### Optimizer Config ###
        ########################
        self.agent_config.policy_optim_cfg.lr_dqn = 1e-4
        self.agent_config.policy_optim_cfg.adam_eps = 1e-6  # default value in pytorch 1e-8

        #################################
        ### Data augmentation config  ###
        #################################
        self.agent_config.policy_hyper_params.aug_class = Torch_Data_Augmentation
        self.agent_config.policy_hyper_params.func_augmentations = [torch_random_shift]



