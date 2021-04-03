import numpy as np
import torch
import torch.nn as nn

from learning_agents.common.common_utils import identity
from learning_agents.architectures.mlp import MLP, GaussianDist, CategoricalDistParams, TanhGaussianDistParams
from learning_agents.utils.tensor_utils import get_device

device = get_device()


class CNNLayer(nn.Module):
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
        super(CNNLayer, self).__init__()
        self._cnn = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.pre_activation_fn = pre_activation_fn
        self.activation_fn = activation_fn
        self.post_activation_fn = post_activation_fn

    def tie_weights(self, source):
        assert type(self._cnn) == type(source)
        self._cnn.weight = source.weight
        self._cnn.bias = source.bias

    def forward(self, x):
        x = self._cnn(x)
        x = self.pre_activation_fn(x)
        x = self.activation_fn(x)
        x = self.post_activation_fn(x)
        return x


class CNN(nn.Module):
    """ Baseline of Convolution neural network. """

    def __init__(self, cnn_layers, fc_layers):
        """
        cnn_layers: ModuleList
        fc_layers: MLP
        """
        super(CNN, self).__init__()

        self._cnn = cnn_layers
        self._fc_layers = fc_layers

        self._cnn = nn.Sequential()
        for i, cnn_layer in enumerate(cnn_layers):
            self._cnn.add_module("cnn_{}".format(i), cnn_layer)

    def get_cnn_features(self, x, is_flatten=True):
        """
        Get the output of CNN.
        """
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        x = self._cnn(x)
        # flatten x
        if is_flatten:
            x = x.view(x.size(0), -1)
        return x

    def forward(self, x, **fc_kwargs):
        """
        Forward method implementation.
        x: torch.Tensor
        :return: torch.Tensor
        """
        x = self.get_cnn_features(x)
        x = self._fc_layers(x, **fc_kwargs)
        return x


class Conv2d_Encoder(nn.Module):
    """ Default CNN Encoder """

    def __init__(self,
                 # output arguments
                 feature_dim,
                 fc_input_size,
                 # conv2d layer arguments
                 input_channels,
                 # conv2d optional arguments
                 channels=(32, 64, 64),
                 kernel_sizes=(8, 4, 3),
                 strides=(4, 2, 1),
                 paddings=(0, 1, 1),
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # post processing
                 has_head=True,
                 output_logits=True
                 ):
        super(Conv2d_Encoder, self).__init__()
        self.output_logits = output_logits
        self.has_head = has_head

        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        self._conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                      kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                             for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides,
                                                                      paddings, activation_fns, post_activation_fns)]
        # noinspection PyTypeChecker
        self._cnn_encoder = CNN(cnn_layers=self._conv_layers, fc_layers=None)
        # post processing functions
        if has_head:
            self._head = nn.Sequential(
                nn.Linear(fc_input_size, feature_dim),
                nn.LayerNorm(feature_dim))

    def tie_conv_layers(self, source):
        assert type(self) == type(source)
        for i in range(len(self._conv_layers)):
            # noinspection PyProtectedMember
            self._conv_layers[i].tie_weights(source._conv_layers[i])

    def forward(self, x):
        features = self._cnn_encoder.get_cnn_features(x)
        if self.has_head:
            features = self._head(features)
        if self.output_logits:
            features = torch.tanh(features)
        return features


class Conv2d_MLP_Model(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
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
                 fc_output_activation=identity
                 ):
        super(Conv2d_MLP_Model, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        # noinspection PyTypeChecker
        fc_layers = MLP(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation,
            output_activation=fc_output_activation
        )

        self._conv_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x):
        return self._conv_mlp.forward(x)

    def get_cnn_features(self, x):
        return self._conv_mlp.get_cnn_features(x, is_flatten=True)

    # noinspection PyProtectedMember
    def get_fc_output(self, cnn_features):
        x = self._conv_mlp._fc_layers(cnn_features)
        return x


class Conv2d_MLP_Gaussian(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=(32, 64, 64),
                 kernel_sizes=(8, 4, 3),
                 strides=(4, 2, 1),
                 paddings=(0, 1, 1),
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=(512,),
                 fc_hidden_activation=torch.relu
                 ):
        super(Conv2d_MLP_Gaussian, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = GaussianDist(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation
        )

        self._conv_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x):
        return self._conv_mlp.forward(x)


class Conv2d_MLP_Categorical(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=(32, 64, 64),
                 kernel_sizes=(8, 4, 3),
                 strides=(4, 2, 1),
                 paddings=(0, 1, 1),
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=(512,),
                 fc_hidden_activation=torch.relu
                 ):
        super(Conv2d_MLP_Categorical, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = CategoricalDistParams(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation
        )

        self._conv_categorical_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x, deterministic=False):
        return self._conv_categorical_mlp.forward(x, deterministic=deterministic)


class Conv2d_MLP_TanhGaussian(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=(32, 64, 64),
                 kernel_sizes=(8, 4, 3),
                 strides=(4, 2, 1),
                 paddings=(0, 1, 1),
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=(512,),
                 fc_hidden_activation=torch.relu
                 ):
        super(Conv2d_MLP_TanhGaussian, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = TanhGaussianDistParams(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation
        )

        self._conv_tanh_gaussian_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x, epsilon=1e-6, deterministic=False, reparameterize=True):
        return self._conv_tanh_gaussian_mlp.forward(x, epsilon=epsilon,
                                                    deterministic=deterministic, reparameterize=reparameterize)


class Conv2d_Flatten_MLP(Conv2d_MLP_Model):
    """
    Augmented convolution neural network, in which a feature vector will be appended to
        the features extracted by CNN before entering mlp
    """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=(32, 64, 64),
                 kernel_sizes=(8, 4, 3),
                 strides=(4, 2, 1),
                 paddings=(0, 1, 1),
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=(100, 100),
                 fc_hidden_activation=torch.relu,
                 fc_output_activation=identity
                 ):
        super(Conv2d_Flatten_MLP, self).__init__(input_channels=input_channels,
                                                 fc_input_size=fc_input_size,
                                                 fc_output_size=fc_output_size,
                                                 channels=channels, kernel_sizes=kernel_sizes, strides=strides,
                                                 paddings=paddings, nonlinearity=nonlinearity,
                                                 use_maxpool=use_maxpool, fc_hidden_sizes=fc_hidden_sizes,
                                                 fc_hidden_activation=fc_hidden_activation,
                                                 fc_output_activation=fc_output_activation)

    def forward(self, *args):
        obs_x, augment_features = args
        cnn_features = self._conv_mlp.get_cnn_features(obs_x)
        features = torch.cat((cnn_features, augment_features), dim=1)
        # noinspection PyProtectedMember
        return self._conv_mlp._fc_layers(features)
