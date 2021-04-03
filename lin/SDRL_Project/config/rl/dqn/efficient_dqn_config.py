from config.rl.dqn.default_dqn_config import DQN_Config


class Efficient_DQN_Config(DQN_Config):
    def __init__(self):
        # simply use the default config here
        DQN_Config.__init__(self)

    def get_agent_config(self):
        DQN_Config.get_agent_config(self)

        # Double Q-Learning
        self.agent_config.policy_hyper_params.use_double_q_update = True

        """
        # N-Step Buffer
        self.agent_config.policy_hyper_params.terminate_life_loss = False
        self.agent_config.policy_hyper_params.n_step = 5
        # Epsilon Exploration
        self.agent_config.policy_hyper_params.explore_strategy = 'epsilon-greedy'
        self.agent_config.policy_hyper_params.epsilon_strategy = 'linear-step'
        self.agent_config.policy_hyper_params.use_noisy_net = False
        self.agent_config.policy_hyper_params.max_epsilon = 1.0
        self.agent_config.policy_hyper_params.min_epsilon = 0.01  # open-ai baselines: 0.01
        self.agent_config.policy_hyper_params.epsilon_decay = 2500
        """
        # N-Step Buffer
        self.agent_config.policy_hyper_params.n_step = 5
        # Epsilon Exploration
        self.agent_config.policy_hyper_params.explore_strategy = 'epsilon-greedy'
        self.agent_config.policy_hyper_params.epsilon_strategy = 'exponential-episode'
        self.agent_config.policy_hyper_params.use_noisy_net = False
        self.agent_config.policy_hyper_params.max_epsilon = 1.0
        self.agent_config.policy_hyper_params.min_epsilon = 0.01  # open-ai baselines: 0.01
        self.agent_config.policy_hyper_params.epsilon_decay = 0.95  # default: 0.9995

        ########################
        ### Optimizer Config ###
        ########################
        self.agent_config.policy_optim_cfg.adam_eps = 0.00015  # default value in pytorch 1e-8

