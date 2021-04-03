import torch
import torch.nn as nn

from learning_agents.common.common_utils import identity
from learning_agents.utils.tensor_utils import get_device

device = get_device()


class DeConvLayer(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride=1,
        padding=0,
        pre_activation_fn=identity,
        activation_fn=torch.relu,
        post_activation_fn=identity,
    ):
        super(DeConvLayer, self).__init__()
        self.de_conv = nn.ConvTranspose2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.pre_activation_fn = pre_activation_fn
        self.activation_fn = activation_fn
        self.post_activation_fn = post_activation_fn

    def forward(self, x):
        x = self.de_conv(x)
        x = self.pre_activation_fn(x)
        x = self.activation_fn(x)
        x = self.post_activation_fn(x)
        return x


# noinspection PyDefaultArgument
class DeConv2d_Model(nn.Module):
    def __init__(self,
                 input_channel=64,
                 channels=[64, 32, 4],
                 kernel_sizes=[3, 4, 8],
                 strides=[1, 2, 4],
                 paddings=[0, 0, 1],
                 activation_fn=[torch.relu, torch.relu, torch.sigmoid],
                 post_activation_fn=[identity, identity, identity]):
        super(DeConv2d_Model, self).__init__()

        self.deconv_layers = nn.Sequential()
        for i, _ in enumerate(channels):
            in_channel = input_channel if i == 0 else channels[i-1]
            deconv_layer = DeConvLayer(input_channels=in_channel, output_channels=channels[i],
                                       kernel_size=kernel_sizes[i], stride=strides[i],
                                       padding=paddings[i], activation_fn=activation_fn[i],
                                       post_activation_fn=post_activation_fn[i])
            self.deconv_layers.add_module("deconv_{}".format(i), deconv_layer)

    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        deconv_x = self.deconv_layers.forward(x)
        return deconv_x

