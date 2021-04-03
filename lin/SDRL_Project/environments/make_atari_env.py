import gym

from environments.wrappers.atari_dopamine_wrapper import DopamineAtariPreprocessor
from environments.wrappers.gym_img_wrapper import GymImageWrapper
from environments.wrappers.frame_stack_wrapper import FrameStack


def make_atari_env(env_name, agent_config):
    """
    Return gym atari environment
    :param env_name: env name
    :param agent_config: an addict
    """
    if 'NoFrameskip' in env_name:
        return FrameStack(DopamineAtariPreprocessor(gym.make(env_name),
                                                    is_stochastic=agent_config.sys_args.env_stochastic,
                                                    frame_skip=agent_config.sys_args.frame_skip,
                                                    screen_size=agent_config.sys_args.frame_size,
                                                    is_normalize=agent_config.sys_args.frame_normalize),
                          num_stack=agent_config.sys_args.frame_stack)
    else:
        return FrameStack(GymImageWrapper(gym.make(env_name),
                                          screen_size=agent_config.sys_args.frame_size,
                                          is_normalize=agent_config.sys_args.frame_normalize),
                          num_stack=agent_config.sys_args.frame_stack)
