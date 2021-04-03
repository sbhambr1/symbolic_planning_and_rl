import numpy as np
import torch
import torch.optim as optim

from learning_agents.architectures.cnn import Conv2d_MLP_Model
from torch.nn.utils import clip_grad_norm_
from learning_agents.rl.value_based_agent.dqn import dqn_utils
from learning_agents.common import common_utils
from learning_agents.rl.value_based_agent.value_based_agent import Value_Based_Agent
from learning_agents.utils.tensor_utils import get_device

device = get_device()


class DQN_Agent(Value_Based_Agent):
    """
    DQN Agent. (Here we assume that we are using stacked frames and grayscaled observation)
    Attribute:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        memory (PrioritizedReplayBuffer): replay memory
        dqn (nn.Module): actor model to select actions
        dqn_target (nn.Module): target actor model to select actions
        dqn_optim (Optimizer): optimizer for training actor
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step number
        episode_step (int): step number of the current episode
        i_episode (int): current episode number
        epsilon (float): parameter for epsilon greedy controller_policy
        n_step_buffer (deque): n-size buffer to calculate n-step returns
        per_beta (float): beta parameter for prioritized replay buffer
        use_n_step (bool): whether or not to use n-step returns
    """

    def __init__(self, env, args, hyper_params, network_cfg, optim_cfg, logger=None):
        """Initialize."""
        Value_Based_Agent.__init__(self, env, args, hyper_params, network_cfg, optim_cfg, logger=logger)

        self._init_network()

    # pylint: disable=attribute-defined-outside-init
    def _init_network(self):
        """Initialize networks and optimizers."""
        self.dqn = Conv2d_MLP_Model(input_channels=self.state_channel,
                                    fc_input_size=self.network_cfg.fc_input_size,
                                    fc_output_size=self.action_dim,
                                    nonlinearity=self.network_cfg.nonlinearity,
                                    channels=self.network_cfg.channels,
                                    kernel_sizes=self.network_cfg.kernel_sizes,
                                    strides=self.network_cfg.strides,
                                    paddings=self.network_cfg.paddings,
                                    fc_hidden_sizes=self.network_cfg.fc_hidden_sizes,
                                    fc_hidden_activation=self.network_cfg.fc_hidden_activation).to(device)
        self.dqn_target = Conv2d_MLP_Model(input_channels=self.state_channel,
                                           fc_input_size=self.network_cfg.fc_input_size,
                                           fc_output_size=self.action_dim,
                                           nonlinearity=self.network_cfg.nonlinearity,
                                           channels=self.network_cfg.channels,
                                           kernel_sizes=self.network_cfg.kernel_sizes,
                                           strides=self.network_cfg.strides,
                                           paddings=self.network_cfg.paddings,
                                           fc_hidden_sizes=self.network_cfg.fc_hidden_sizes,
                                           fc_hidden_activation=self.network_cfg.fc_hidden_activation).to(device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        for param in self.dqn_target.parameters():
            param.requires_grad = False

        # create optimizer
        self.dqn_optim = optim.Adam(
            self.dqn.parameters(),
            lr=self.optim_cfg.lr_dqn,
            weight_decay=self.optim_cfg.weight_decay,
            eps=self.optim_cfg.adam_eps,
        )

        # init network from file
        self._init_from_file()
        if not self.testing and self.logger is not None:
            self.logger.watch_wandb([self.dqn, self.dqn_target])

    def _init_from_file(self):
        # load the optimizer and model parameters
        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

    def select_action(self, state):
        """Select an action from the input space."""
        # epsilon greedy controller_policy
        # pylint: disable=comparison-with-callable
        self.epsilon = self.explore_strategy.get_epsilon(self.total_step, self.i_episode)
        if not self.testing and \
                (self.epsilon > np.random.random() or self.total_step < self.hyper_params.init_random_actions):
            selected_action = np.array(self.env.action_space.sample())
        else:
            state = self._preprocess_state(state)
            self.dqn.eval()
            with torch.no_grad():
                selected_action = self.dqn(state).argmax(1).squeeze()
            self.dqn.train()
            selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    def _get_dqn_loss(self, experiences, gamma):
        """Return element-wise dqn loss and Q-values."""
        return dqn_utils.calculate_dqn_loss(
            model=self.dqn,
            target_model=self.dqn_target,
            experiences=experiences,
            gamma=gamma,
            use_double_q_update=self.hyper_params.use_double_q_update,
            reward_clip=self.hyper_params.reward_clip,
            reward_scale=self.hyper_params.reward_scale,
        )

    # noinspection PyMethodMayBeStatic
    def _preprocess_experience(self, experiences, weights):
        return experiences, weights

    def _sample_experiences_one_step(self):
        n_sample = min(len(self.memory), self.hyper_params.batch_size)
        if self.use_prioritized:
            experiences_one_step = self.memory.sample(self.per_beta)
            weights, indices = experiences_one_step[-3:-1]
            indices = np.array(indices)
            # re-normalize the weights such that they sum up to the value of batch_size
            weights = weights / torch.sum(weights) * float(n_sample)
        else:
            indices = np.random.choice(len(self.memory), size=n_sample, replace=False)
            weights = torch.from_numpy(np.ones(shape=(indices.shape[0], 1))).float().to(device)
            experiences_one_step = self.memory.sample(indices=indices)
        return experiences_one_step, indices, weights

    def update_model(self):
        """Train the model after each episode."""
        # 1 step loss
        experiences_one_step, indices, weights = self._sample_experiences_one_step()
        experiences_one_step, sample_weights = self._preprocess_experience(experiences_one_step, weights)
        dqn_loss_element_wise, q_values = self._get_dqn_loss(experiences_one_step, self.hyper_params.gamma)
        dqn_loss = torch.mean(dqn_loss_element_wise * sample_weights)

        # n step loss
        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices)
            experiences_n, sample_weights = self._preprocess_experience(experiences_n, weights)
            gamma = self.hyper_params.gamma ** self.hyper_params.n_step
            dq_loss_n_element_wise, q_values_n = self._get_dqn_loss(experiences_n, gamma)

            # to update loss and priorities
            q_values = 0.5 * (q_values + q_values_n)
            # mix of 1-step and n-step returns
            dqn_loss_element_wise += dq_loss_n_element_wise * self.hyper_params.w_n_step
            dqn_loss = torch.mean(dqn_loss_element_wise * sample_weights)

        # total loss
        loss = dqn_loss

        # q_value regularization (not used when w_q_reg is set to 0)
        if self.optim_cfg.w_q_reg > 0:
            q_regular = torch.norm(q_values, 2).mean() * self.optim_cfg.w_q_reg
            loss = loss + q_regular

        self.dqn_optim.zero_grad()
        loss.backward()
        if self.hyper_params.gradient_clip is not None:
            clip_grad_norm_(self.dqn.parameters(), self.hyper_params.gradient_clip)
        self.dqn_optim.step()

        # update target networks
        if self.total_step % self.hyper_params.target_update_freq == 0:
            common_utils.soft_update(self.dqn, self.dqn_target, self.hyper_params.tau)

        # update priorities in PER
        if self.use_prioritized:
            loss_for_prior = dqn_loss_element_wise.detach().cpu().numpy().squeeze()[:len(indices)]
            new_priorities = loss_for_prior + self.hyper_params.per_eps
            if (new_priorities <= 0).any().item():
                print('[ERROR] new priorities less than 0. Loss info: ', str(loss_for_prior))

            # noinspection PyUnresolvedReferences
            self.memory.update_priorities(indices, new_priorities)

            # increase beta
            fraction = min(float(self.total_step) / self.args.max_step, 1.0)
            self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

        # whether to use noise net
        if self.hyper_params.use_noisy_net:
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()

        return loss.item(), q_values.mean().item()

    def load_params(self, path):
        """Load model and optimizer parameters."""
        Value_Based_Agent.load_params(self, path)

        params = torch.load(path, map_location=device)
        self.dqn.load_state_dict(params["dqn_state_dict"])
        self.dqn_target.load_state_dict(params["dqn_target_state_dict"])
        self.dqn_optim.load_state_dict(params["dqn_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_step):
        """Save model and optimizer parameters."""
        params = {
            "dqn_state_dict": self.dqn.state_dict(),
            "dqn_target_state_dict": self.dqn_target.state_dict(),
            "dqn_optim_state_dict": self.dqn_optim.state_dict(),
        }

        if self.logger is not None:
            self.logger.save_models(params, prefix='model', postfix=str(n_step), is_snapshot=True)

    def write_log(self, log_value):
        """Write log about loss and score"""
        loss, score, avg_time_cost, avg_score_window = log_value
        log_info = {
            "episode": self.i_episode,
            "score": score,
            "episode step": self.episode_step,
            "total step": self.total_step,
            "epsilon": self.explore_strategy.get_epsilon(self.total_step, self.i_episode),
            "dqn loss": loss[0],
            "avg q values": loss[1],
            "time per each step": avg_time_cost,
            "avg score window": avg_score_window,
        }

        print("[INFO] %s\n" % (str(log_info),))
        if self.logger is not None:
            self.logger.log_wandb(log_info, step=self.total_step)

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain(self):
        """Pretraining steps."""
        pass

    def do_post_episode_update(self, *argv):
        pass

