from config.rl.dqn.default_dqn_config import DQN_Config


class SQIL_DQN_Config(DQN_Config):
    def __init__(self):
        # simply use the default config here
        DQN_Config.__init__(self)

    def get_agent_config(self):
        DQN_Config.get_agent_config(self)
        # Override config
        self.agent_config.policy_hyper_params.n_step = 1
        self.agent_config.policy_hyper_params.batch_size = 64
        self.agent_config.policy_hyper_params.reward_freq = None
        self.agent_config.policy_hyper_params.demo_reward = 0.01
        self.agent_config.policy_hyper_params.use_prioritized = False
        self.agent_config.policy_hyper_params.use_double_q_update = False

        ############################
        ## Hyper Parameter Config ##
        ############################
        self.agent_config.policy_hyper_params.train_test_split = 1.0

        ########################
        ### Optimizer Config ###
        ########################
        self.agent_config.policy_optim_cfg.lr_dqn = 1e-4
        self.agent_config.policy_optim_cfg.adam_eps = 1e-6  # default value in pytorch 1e-8



