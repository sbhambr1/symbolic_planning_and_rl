import torch
from addict import Dict

from config.agent_config import Agent_Config


class Nature_DQN_Config(Agent_Config):
    def __init__(self):
        Agent_Config.__init__(self)

    def get_agent_config(self):
        ############################
        ## Hyper Parameter Config ##
        ############################
        policy_hyper_params = Dict()
        self.agent_config.policy_hyper_params = policy_hyper_params
        # hyper params
        policy_hyper_params.gamma = 0.99
        policy_hyper_params.tau = 5e-3   # for soft update: 5e-3, otherwise: 1
        policy_hyper_params.target_update_freq = 1       # for soft update: 1, otherwise: 8000
        policy_hyper_params.buffer_size = int(1e5)
        policy_hyper_params.batch_size = 32
        policy_hyper_params.init_random_actions = int(1e3)
        policy_hyper_params.update_starts_from = int(1e3)      # open-ai baselines default 1e4
        policy_hyper_params.policy_multiple_update = 1     # multiple training updates
        policy_hyper_params.train_freq = 4     # default: 4
        policy_hyper_params.reward_clip = (-1, 1)
        policy_hyper_params.reward_scale = 1.0
        policy_hyper_params.gradient_clip = None
        policy_hyper_params.terminate_life_loss = False
        # N-Step Buffer
        policy_hyper_params.n_step = 1  # if n_step <= 1, use common replay buffer otherwise n_step replay buffer
        policy_hyper_params.w_n_step = 1.0  # n-step loss weight
        # Double Q-Learning
        policy_hyper_params.use_double_q_update = False
        # Prioritized Replay Buffer
        policy_hyper_params.use_prioritized = False
        policy_hyper_params.per_alpha = 0.6  # open-ai baselines default: 0.6, alpha -> 1, full prioritization
        policy_hyper_params.per_beta = 0.4  # beta can start small (for stability concern and anneals towards 1)
        policy_hyper_params.per_eps = 1e-6
        # Noisy Net
        policy_hyper_params.use_noisy_net = False
        policy_hyper_params.std_init = 0.5
        # Epsilon Greedy
        policy_hyper_params.explore_strategy = 'epsilon-greedy'
        policy_hyper_params.epsilon_strategy = 'exponential-episode'
        policy_hyper_params.max_epsilon = 1.0
        policy_hyper_params.min_epsilon = 0.01  # open-ai baselines: 0.01
        policy_hyper_params.epsilon_decay = 0.9     # default: 0.9995

        #################################
        ## Network Architecture Config ##
        #################################
        policy_network_cfg = Dict()
        self.agent_config.policy_network_cfg = policy_network_cfg
        # CNN
        policy_network_cfg.nonlinearity = torch.relu
        policy_network_cfg.channels = [32, 64, 64]
        policy_network_cfg.kernel_sizes = [8, 4, 3]
        policy_network_cfg.strides = [4, 2, 1]
        policy_network_cfg.paddings = [0, 0, 0]
        # FC
        policy_network_cfg.fc_input_size = 3136
        policy_network_cfg.fc_hidden_sizes = [512]
        policy_network_cfg.fc_hidden_activation = torch.relu

        ########################
        ### Optimizer Config ###
        ########################
        policy_optim_cfg = Dict()
        self.agent_config.policy_optim_cfg = policy_optim_cfg
        policy_optim_cfg.lr_dqn = 1e-4
        policy_optim_cfg.adam_eps = 1e-8  # default value in pytorch 1e-8
        policy_optim_cfg.weight_decay = 0
        policy_optim_cfg.w_q_reg = 0  # use q_value regularization

