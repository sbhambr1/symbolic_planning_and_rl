from environments.planning_envs.montezuma.montezuma_planning_env import Montezuma_Planning_Env


def make_planning_env(env_name, agent_config):
    env = None
    if 'montezuma' in env_name.lower():
        env = Montezuma_Planning_Env(env_name, agent_config)
    return env
