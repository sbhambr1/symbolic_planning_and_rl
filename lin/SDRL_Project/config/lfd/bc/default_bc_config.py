from config.rl.dqn.nature_dqn_config import Nature_DQN_Config


class Default_BC_Config(Nature_DQN_Config):
    def __init__(self):
        # simply use the default config here
        Nature_DQN_Config.__init__(self)

    def get_agent_config(self):
        Nature_DQN_Config.get_agent_config(self)
        # Override config
        self.agent_config.policy_hyper_params.n_step = 1
        self.agent_config.policy_hyper_params.use_prioritized = False
        self.agent_config.policy_hyper_params.gradient_clip = None

        self.agent_config.policy_hyper_params.policy_save_freq = 200

        ############################
        ## Hyper Parameter Config ##
        ############################
        self.agent_config.policy_hyper_params.action_mechanism = 'softmax'  # options: 'softmax' and 'deterministic'
        self.agent_config.policy_hyper_params.batch_size = 32
        self.agent_config.policy_hyper_params.n_eval_rollouts = 5   # number of evaluation rollout
        self.agent_config.policy_hyper_params.train_test_split = 1.0

        ########################
        ### Optimizer Config ###
        ########################
        self.agent_config.policy_optim_cfg.lr_dqn = 1e-4
        self.agent_config.policy_optim_cfg.weight_decay = 1e-5


