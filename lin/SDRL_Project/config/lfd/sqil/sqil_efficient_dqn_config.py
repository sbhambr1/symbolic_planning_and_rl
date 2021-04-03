from config.rl.dqn.efficient_dqn_config import Efficient_DQN_Config


class SQIL_Efficient_DQN_Config(Efficient_DQN_Config):
    def __init__(self):
        # simply use the default config here
        Efficient_DQN_Config.__init__(self)

    def get_agent_config(self):
        Efficient_DQN_Config.get_agent_config(self)
        # Override config
        self.agent_config.policy_hyper_params.n_step = 1
        self.agent_config.policy_hyper_params.batch_size = 64
        self.agent_config.policy_hyper_params.reward_freq = None
        self.agent_config.policy_hyper_params.demo_reward = 0.01

        ############################
        ## Hyper Parameter Config ##
        ############################
        self.agent_config.policy_optim_cfg.lr_dqn = 1e-4
        self.agent_config.policy_optim_cfg.adam_eps = 0.00015  # default value in pytorch 1e-8


