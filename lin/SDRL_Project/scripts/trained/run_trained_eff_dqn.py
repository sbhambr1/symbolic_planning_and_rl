from config.agent_config import Agent_Config
from config.rl.dqn.efficient_dqn_config import Efficient_DQN_Config
from environments.make_env import make_env
from learning_agents.common.eval_agent import eval_agent
from learning_agents.rl.value_based_agent.dqn.dqn_agent import DQN_Agent
from learning_agents.rl.value_based_agent.dqn.dueling_dqn_agent import Dueling_DQN_Agent
from oracles.utils.model_utils import get_model
from utils.info_displayer import InfoDisplayer

env_name = 'taxi'
model_path = '/Users/lguan/Desktop/EXPAND/model_0_0.tar'


def run():
    trained_model = get_model(env_name,
                              config_class=Efficient_DQN_Config,
                              model_path=model_path,
                              model_class=Dueling_DQN_Agent)
    env_config = Efficient_DQN_Config()
    env_config.get_agent_config()
    env = make_env(env_name, env_config.agent_config)
    info_displayer = InfoDisplayer(screen_height=150 * 6, screen_width=250, frame_time=0.05)
    eval_agent(env, trained_model, 5, verbose=True, render=True,
               info_displayer=info_displayer)
    env.close()


if __name__ == '__main__':
    run()
