from config.lfd.bc.default_bc_config import Default_BC_Config
from config.rl.dqn.default_dqn_config import DQN_Config
from environments.make_env import make_env
from learning_agents.common.eval_agent import eval_agent
from learning_agents.lfd.behavioural_cloning.bc_agent import BC_Agent
from oracles.utils.model_utils import get_model, get_lfd_model
from utils.info_displayer import InfoDisplayer

env_name = 'enduro'
model_path = '/Users/lguan/Desktop/EXPAND/model_400_0.tar'


def run():
    trained_model = get_lfd_model(env_name,
                                  config_class=Default_BC_Config,
                                  model_path=model_path,
                                  model_class=BC_Agent)
    env_config = Default_BC_Config()
    env_config.get_agent_config()
    env = make_env(env_name, env_config.agent_config)
    info_displayer = InfoDisplayer(screen_height=150 * 6, screen_width=250, frame_time=0.05)
    eval_agent(env, trained_model, 5, verbose=True, render=True,
               info_displayer=info_displayer)
    env.close()


if __name__ == '__main__':
    run()
