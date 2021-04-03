import torch

from learning_agents.architectures.cnn import Conv2d_MLP_Model
from learning_agents.common.common_utils import identity


class Nature_DQN_Model(Conv2d_MLP_Model):
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size):
        Conv2d_MLP_Model.__init__(self, input_channels=input_channels, fc_input_size=fc_input_size,
                                  fc_output_size=fc_output_size,
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
                                  fc_output_activation=identity)

