from config.lfd.bc.default_bc_config import Default_BC_Config


class Default_Agil_Config(Default_BC_Config):
    def __init__(self):
        # simply use the default config here
        Default_BC_Config.__init__(self)

    def get_agent_config(self):
        Default_BC_Config.get_agent_config(self)
        ########################
        ### Optimizer Config ###
        ########################
        self.agent_config.policy_optim_cfg.lr_dqn = 0.001
        self.agent_config.policy_optim_cfg.weight_decay = 0.00001


