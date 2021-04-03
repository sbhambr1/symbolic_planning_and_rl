from config.rl.dqn.nature_dqn_config import Nature_DQN_Config


class DQN_Config(Nature_DQN_Config):
    def __init__(self):
        Nature_DQN_Config.__init__(self)

    def get_agent_config(self):
        Nature_DQN_Config.get_agent_config(self)

        ############################
        ## Hyper Parameter Config ##
        ############################
        self.agent_config.policy_hyper_params.gradient_clip = 10.0
        # N-Step Buffer
        self.agent_config.policy_hyper_params.n_step = 5
        # Double Q-Learning
        self.agent_config.policy_hyper_params.use_double_q_update = False
        # Prioritized Replay Buffer
        self.agent_config.policy_hyper_params.use_prioritized = True
        # open-ai baselines default: 0.6, alpha -> 1, full prioritization
        self.agent_config.policy_hyper_params.per_alpha = 0.6
        # beta can start small (for stability concern and anneals towards 1)
        self.agent_config.policy_hyper_params.per_beta = 0.4
        self.agent_config.policy_hyper_params.per_eps = 1e-6

        ########################
        ### Optimizer Config ###
        ########################
        self.agent_config.policy_optim_cfg.lr_dqn = 1e-4
        self.agent_config.policy_optim_cfg.adam_eps = 1e-6  # default value in pytorch 1e-8


