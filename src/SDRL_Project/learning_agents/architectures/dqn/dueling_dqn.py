import torch
import torch.nn as nn

from learning_agents.architectures.cnn import Conv2d_Encoder
from learning_agents.architectures.mlp import MLP
from learning_agents.common.common_utils import identity


class Dueling_DQN_Model(nn.Module):
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 action_dim,
                 # conv2d optional arguments
                 channels=(32, 64, 64),
                 kernel_sizes=(8, 4, 3),
                 strides=(4, 2, 1),
                 paddings=(0, 1, 1),
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=(512,),
                 fc_hidden_activation=torch.relu,
                 fc_output_activation=identity):
        super(Dueling_DQN_Model, self).__init__()
        self.action_dim = action_dim
        # init cnn encoder
        self._encoder = Conv2d_Encoder(input_channels=input_channels,
                                       # conv2d optional arguments
                                       channels=channels,
                                       kernel_sizes=kernel_sizes,
                                       strides=strides,
                                       paddings=paddings,
                                       nonlinearity=nonlinearity,
                                       use_maxpool=use_maxpool,
                                       # post processing
                                       has_head=False,
                                       output_logits=False,
                                       # output arguments
                                       feature_dim=fc_input_size,  # not used here
                                       fc_input_size=fc_input_size,  # not used here
                                       )

        self._adv_head = MLP(
            input_size=fc_input_size,
            output_size=action_dim,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation,
            output_activation=fc_output_activation
        )

        self._val_head = MLP(
            input_size=fc_input_size,
            output_size=1,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation,
            output_activation=fc_output_activation
        )

    def forward(self, x):
        cnn_features = self.get_cnn_features(x)
        return self.get_fc_output(cnn_features)

    def get_cnn_features(self, x):
        return self._encoder(x)

    def get_fc_output(self, cnn_features):
        advantages = self._adv_head(cnn_features)  # advantage value
        state_values = self._val_head(cnn_features).expand(cnn_features.size(0), self.action_dim)  # state value
        # compute q values
        q_values = state_values + advantages - advantages.mean(dim=-1, keepdim=True).expand(state_values.size(0),
                                                                                            self.action_dim)
        return q_values
