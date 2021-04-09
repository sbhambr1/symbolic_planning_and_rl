import os
import sys

from config.planning_rl.sdrl.sdrl_config import SDRL_Config
from learning_agents.planning_rl.sdrl.sdrl_agent import SDRL_Agent
from utils.experiment_manager import init_experiment

experiment_log_dir = 'tmp/'
model_name = 'sdrl'


def run():
    expr_config = SDRL_Config()
    expr_config.get_agent_config()

    sys_args = expr_config.agent_config.sys_args
    policy_hyper_params = expr_config.agent_config.policy_hyper_params
    policy_network_cfg = expr_config.agent_config.policy_network_cfg
    policy_optim_cfg = expr_config.agent_config.policy_optim_cfg

    files_to_save = [os.path.abspath(sys.modules[SDRL_Agent.__module__].__file__)]
    env, expr_manager = init_experiment(env_name=sys_args.env,
                                        model_name=model_name,
                                        env_hyper_params=expr_config.agent_config,
                                        random_seed=sys_args.seed,
                                        files_to_save=files_to_save, config_to_save=expr_config.agent_config,
                                        experiment_log_dir=experiment_log_dir,
                                        use_wandb=sys_args.use_wandb, virtual_display=sys_args.virtual_display)

    sdrl_agent = SDRL_Agent(env, sys_args, policy_hyper_params,
                            policy_network_cfg, policy_optim_cfg, logger=expr_manager)
    if sys_args.test:
        sdrl_agent.test()
    else:
        sdrl_agent.train()


if __name__ == '__main__':
    run()
