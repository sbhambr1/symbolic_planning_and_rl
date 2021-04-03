from config.human_study.default_human_config import Default_Human_Config
from config.rl.dqn.default_dqn_config import DQN_Config


class DQN_TAMER_Config(DQN_Config, Default_Human_Config):
    def __init__(self):
        DQN_Config.__init__(self)
        Default_Human_Config.__init__(self)

    def get_agent_config(self):
        DQN_Config.get_agent_config(self)
        Default_Human_Config.get_agent_config(self)

        ############################
        ## Hyper Parameter Config ##
        ############################
        self.agent_config.policy_hyper_params.batch_size = 64

        # Epsilon Greedy
        self.agent_config.policy_hyper_params.explore_strategy = 'epsilon-greedy'
        self.agent_config.policy_hyper_params.epsilon_strategy = 'exponential-episode'
        self.agent_config.policy_hyper_params.max_epsilon = 1.0
        self.agent_config.policy_hyper_params.min_epsilon = 0.01  # open-ai baselines: 0.01
        self.agent_config.policy_hyper_params.epsilon_decay = 0.9  # default: 0.9995

        self.agent_config.policy_hyper_params.alpha_human = 1.0
        self.agent_config.policy_hyper_params.human_feedback_multiple_update = 1
        self.agent_config.policy_hyper_params.feedback_batch_size = 64
        self.agent_config.policy_hyper_params.feedback_buffer_size = int(5e4)

        self.agent_config.policy_optim_cfg.weight_decay = 1e-8



