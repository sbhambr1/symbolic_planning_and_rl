from config.lfd.dqfd.dqfd_config import DQfD_Config


class Efficient_DQfD_Config(DQfD_Config):
    def __init__(self):
        # simply use the default config here
        DQfD_Config.__init__(self)

    def get_agent_config(self):
        DQfD_Config.get_agent_config(self)

        self.agent_config.policy_hyper_params.use_double_q_update = True

        ########################
        ### Optimizer Config ###
        ########################
        self.agent_config.policy_optim_cfg.adam_eps = 0.00015  # default value in pytorch 1e-8



