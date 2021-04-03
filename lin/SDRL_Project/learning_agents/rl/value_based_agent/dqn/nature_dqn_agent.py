from torch import optim

from learning_agents.architectures.dqn.nature_dqn import Nature_DQN_Model
from learning_agents.rl.value_based_agent.dqn.dqn_agent import DQN_Agent
from learning_agents.utils.tensor_utils import get_device

device = get_device()


class Nature_DQN_Agent(DQN_Agent):
    def __init__(self, env, args, hyper_params, network_cfg, optim_cfg, logger=None):
        """Initialize."""
        DQN_Agent.__init__(self, env, args, hyper_params, network_cfg, optim_cfg, logger=logger)

    def _init_network(self):
        self.dqn = Nature_DQN_Model(input_channels=self.state_channel,
                                    fc_input_size=self.network_cfg.fc_input_size,
                                    fc_output_size=self.action_dim).to(device)
        self.dqn_target = Nature_DQN_Model(input_channels=self.state_channel,
                                           fc_input_size=self.network_cfg.fc_input_size,
                                           fc_output_size=self.action_dim).to(device)
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
