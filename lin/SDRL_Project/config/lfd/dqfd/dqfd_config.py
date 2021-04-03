from config.rl.dqn.default_dqn_config import DQN_Config


class DQfD_Config(DQN_Config):
    def __init__(self):
        # simply use the default config here
        DQN_Config.__init__(self)

    def get_agent_config(self):
        DQN_Config.get_agent_config(self)

        self.agent_config.policy_hyper_params.tau = 5e-3  # for soft update: 5e-3
        self.agent_config.policy_hyper_params.target_update_freq = 1  # for soft update: 1
        # Override config
        self.agent_config.policy_hyper_params.init_random_actions = int(0)
        self.agent_config.policy_hyper_params.update_starts_from = int(0)  # open-ai baselines default 1e4
        self.agent_config.policy_hyper_params.n_step = 5
        self.agent_config.policy_hyper_params.batch_size = 64
        self.agent_config.policy_hyper_params.use_double_q_update = True
        self.agent_config.policy_hyper_params.use_prioritized = True
        # open-ai baselines default: 0.6, alpha -> 1, full prioritization
        self.agent_config.policy_hyper_params.per_alpha = 0.4
        # beta can start small (for stability concern and anneals towards 1)
        self.agent_config.policy_hyper_params.per_beta = 0.6
        self.agent_config.policy_hyper_params.per_eps = 1e-3
        self.agent_config.policy_hyper_params.per_eps_demo = 1.0,  # default: 1.0
        # LfD
        self.agent_config.policy_hyper_params.margin = 0.8
        self.agent_config.policy_hyper_params.pretrain_verbose_freq = 500
        self.agent_config.policy_hyper_params.lambda1 = 1.0  # n-step return weight
        self.agent_config.policy_hyper_params.lambda2 = 1.0  # supervised loss weight
        self.agent_config.policy_hyper_params.min_lambda2 = 0.001
        self.agent_config.policy_hyper_params.lambda2_decay = 25000
        # Epsilon Greedy
        self.agent_config.policy_hyper_params.explore_strategy = 'epsilon-greedy'
        self.agent_config.policy_hyper_params.epsilon_strategy = 'exponential-episode'
        self.agent_config.policy_hyper_params.use_noisy_net = False
        self.agent_config.policy_hyper_params.max_epsilon = 1.0
        self.agent_config.policy_hyper_params.min_epsilon = 0.01  # open-ai baselines: 0.01
        self.agent_config.policy_hyper_params.epsilon_decay = 0.95

        ############################
        ## Hyper Parameter Config ##
        ############################
        self.agent_config.policy_hyper_params.train_test_split = 1.0

        ########################
        ### Optimizer Config ###
        ########################
        self.agent_config.policy_optim_cfg.lr_dqn = 1e-4
        self.agent_config.policy_optim_cfg.adam_eps = 1e-6  # default value in pytorch 1e-8
        self.agent_config.policy_optim.weight_decay = 1e-5



