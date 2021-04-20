import os
import sys

from config.rl.dqn.nature_dqn_config import Nature_DQN_Config
from learning_agents.rl.value_based_agent.dqn.nature_dqn_agent import Nature_DQN_Agent
from learning_agents.rl.value_based_agent.trainer_value_based_agent import Train_Value_Based_Agent
from utils.experiment_manager import init_experiment

experiment_log_dir = 'results/'
model_name = 'nature-dqn'


def run():
    expr_config = Nature_DQN_Config()
    expr_config.get_agent_config()

    sys_args = expr_config.agent_config.sys_args
    policy_hyper_params = expr_config.agent_config.policy_hyper_params
    policy_network_cfg = expr_config.agent_config.policy_network_cfg
    policy_optim_cfg = expr_config.agent_config.policy_optim_cfg

    files_to_save = [os.path.abspath(sys.modules[Nature_DQN_Agent.__module__].__file__)]
    env, expr_manager = init_experiment(env_name=sys_args.env,
                                        model_name=model_name,
                                        env_hyper_params=expr_config.agent_config,
                                        random_seed=sys_args.seed,
                                        files_to_save=files_to_save, config_to_save=expr_config.agent_config,
                                        experiment_log_dir=experiment_log_dir,
                                        use_wandb=sys_args.use_wandb, virtual_display=sys_args.virtual_display)
    # create agent
    dqn_agent = Nature_DQN_Agent(env, sys_args, policy_hyper_params,
                                 policy_network_cfg, policy_optim_cfg, logger=expr_manager)
    # create agent trainer
    agent_trainer = Train_Value_Based_Agent(dqn_agent, env)

    if sys_args.test:
        agent_trainer.test()
    else:
        agent_trainer.train()


if __name__ == '__main__':
    run()
