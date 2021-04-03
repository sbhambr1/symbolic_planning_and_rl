import argparse
import time

from addict import Dict


class Agent_Config:
    """
    Abstract config class used for all agents.
    """

    def __init__(self):
        self.agent_config = Dict()
        self.parse_sys_args()

    def parse_sys_args(self):
        """
        Read command line arguments, save into agent config
        :return: ArgumentParser
        """
        parser = argparse.ArgumentParser(description="Parsing command line arguments")

        parser.add_argument("--env", type=str, default=None,
                            help="the name of the environment")
        parser.add_argument("--env-stochastic", action="store_true", default=False,
                            help="set to make the environment stochastic")
        parser.add_argument("--virtual-display", dest="virtual_display", action="store_true", default=False,
                            help="open virtual display")
        parser.add_argument("--seed", type=int, default=int(round(time.time())),
                            help="random seed for reproducibility")
        parser.add_argument("--test", dest="test", action="store_true",
                            help="test mode (no training)")
        parser.add_argument("--load-from", type=str, default=None,
                            help="load the saved model and optimizer at the beginning")
        parser.add_argument("--use-wandb", dest="use_wandb", action="store_true", default=False,
                            help="use wandb for logging")
        parser.add_argument("--debug-info", action="store_true", default=False,
                            help="compute and record debugging info during the training (might be expensive)")
        # env setting
        parser.add_argument("--max-step", type=int, default=500000,
                            help="maximum environment step (default: 500k)")
        parser.add_argument("--max-traj-len", type=int, default=10000,
                            help="maximum step in a single trajectory (default: 10k)")
        parser.add_argument("--frame-size", type=int, default=84,
                            help="frame size")
        parser.add_argument("--frame-stack", type=int, default=4,
                            help="frame stack")
        parser.add_argument("--frame-skip", type=int, default=4,
                            help="frame skip")
        parser.add_argument("--frame-gray", action="store_false", default=True,
                            help="frame to grayscale")
        parser.add_argument("--frame-normalize", action="store_false", default=True,
                            help="normalize frame")
        # if learn from demo
        parser.add_argument("--pretrain-iter", type=int, default=1000,
                            help="number of epochs for pretraining with demos")
        parser.add_argument("--demo-path", type=str, default=None,
                            help="demonstration path if learning from demo")
        # rendering
        parser.add_argument("--render", dest="render", action="store_true", default=False,
                            help="turn on rendering")
        parser.add_argument("--render-after", type=int, default=0,
                            help="start rendering after the input number of episode")
        parser.add_argument('--render-freq', type=int, default=10,
                            help='render frequency (default: 10)')
        # model snapshot
        parser.add_argument("--save-period", type=int, default=100000,
                            help="save model snapshot every k steps")
        # score and evaluation logging
        parser.add_argument("--avg-score-window", dest="avg_score_window", type=int, default=10,
                            help="window size for recording running average score")
        parser.add_argument("--eval-score-window", dest="eval_score_window", type=int, default=10,
                            help="window size for recording running average evaluation score")
        parser.add_argument("--eval-period", type=int, default=5,
                            help="perform an evaluation rollout every k episodes")
        parser.add_argument("--max-eval-step", type=int, default=100000,
                            help="max step in an eval rollout")

        # save system args in agent config
        sys_args = Dict()
        self.agent_config.sys_args = sys_args
        args, unknown = parser.parse_known_args()
        for arg in vars(args):
            sys_args[arg] = getattr(args, arg)

        return parser

    def get_agent_config(self):
        pass
