import argparse

from config.rl.dqn.default_dqn_config import DQN_Config


class SDRL_Config(DQN_Config):
    def __init__(self):
        DQN_Config.__init__(self)

    def parse_sys_args(self):
        """
        Read command line arguments, save into agent config
        :return: ArgumentParser
        """
        super(SDRL_Config, self).parse_sys_args()
        parser = argparse.ArgumentParser(description="Parsing command line arguments")

        parser.add_argument("--model-dir", type=str, default=None,
                            help="The directory under which sub-goal files is located.")

        # save system args in agent config
        args, unknown = parser.parse_known_args()
        for arg in vars(args):
            self.agent_config.sys_args[arg] = getattr(args, arg)

        return parser

    def get_agent_config(self):
        DQN_Config.get_agent_config(self)

        ###################################
        ## Hyper Parameter Config: SDRL  ##
        ###################################
        self.agent_config.policy_hyper_params.max_goal_step = 500

        #########################################
        ## Hyper Parameter Config: Q-Learning  ##
        #########################################
        self.agent_config.policy_hyper_params.gradient_clip = 10.0
        # Double Q-Learning
        self.agent_config.policy_hyper_params.use_double_q_update = False
        # Prioritized Replay Buffer
        self.agent_config.policy_hyper_params.use_prioritized = True
        # open-ai baselines default: 0.6, alpha -> 1, full prioritization
        self.agent_config.policy_hyper_params.per_alpha = 0.6
        # beta can start small (for stability concern and anneals towards 1)
        self.agent_config.policy_hyper_params.per_beta = 0.4
        self.agent_config.policy_hyper_params.per_eps = 1e-6

        # N-Step Buffer
        self.agent_config.policy_hyper_params.terminate_life_loss = False
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
        self.agent_config.policy_optim_cfg.lr_dqn = 1e-4
        self.agent_config.policy_optim_cfg.adam_eps = 1e-6  # default value in pytorch 1e-8


