from gym import envs

from environments.make_planning_env import make_planning_env
from environments.robot_taxi.robot_taxi_env import Robot_Taxi_Env
from environments.wrappers.frame_stack_wrapper import FrameStack
from environments.wrappers.gym_img_wrapper import GymImageWrapper

atari_game_list = [env_spec.id for env_spec in envs.registry.all()]


def make_env(env_name, agent_config):
    env = None
    if env_name == 'taxi':
        env = get_robot_taxi_env(agent_config)
    elif len(env_name) > 5 and 'plan-' == env_name[:5]:
        env_name = env_name[:5]
        # check if the env name is an Atari game
        for atari_game in atari_game_list:
            if env_name.lower() in atari_game.lower():
                env_name = get_atari_name(env_name)
        make_planning_env(env_name, agent_config)
    else:
        for atari_game in atari_game_list:
            if env_name.lower() in atari_game.lower():
                from environments.make_atari_env import make_atari_env
                env = make_atari_env(get_atari_name(env_name), agent_config)
                break
    return env


def get_atari_name(env_name):
    name = env_name[0].upper() + env_name[1:] + 'NoFrameskip-v4'
    print('[INFO] make env - ', name)
    return name


def get_robot_taxi_env(agent_config):
    return FrameStack(GymImageWrapper(Robot_Taxi_Env(),
                                      screen_size=agent_config.sys_args.frame_size,
                                      is_normalize=agent_config.sys_args.frame_normalize),
                      num_stack=agent_config.sys_args.frame_stack)
