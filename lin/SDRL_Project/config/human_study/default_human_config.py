import argparse
import time

from addict import Dict

from config.agent_config import Agent_Config


class Default_Human_Config(Agent_Config):
    def __init__(self):
        super(Default_Human_Config, self).__init__()
        self.agent_config = Dict()
        self.parse_sys_args()

    def parse_sys_args(self):
        """
        Read command line arguments, save into agent config
        :return: ArgumentParser
        """
        super(Default_Human_Config, self).parse_sys_args()
        parser = argparse.ArgumentParser(description="Parsing command line arguments")

        parser.add_argument("--env", type=str, default=None,
                            help="the name of the environment")
        parser.add_argument("--virtual-display", dest="virtual_display", action="store_true", default=False,
                            help="open virtual display")
        parser.add_argument("--seed", type=int, default=int(round(time.time())),
                            help="random seed for reproducibility")
        parser.add_argument("--human", action="store_true", default=False,
                            help="whether to get guidance from human")
        # demo collecting config
        parser.add_argument("--traj-len", type=int, default=5000,
                            help="maximum length of a trajectory (default: 5000)")
        parser.add_argument("--n-episodes", type=int, default=3,
                            help="number of episodes to collect")
        parser.add_argument("--save-video", action="store_true", default=False,
                            help="whether to save the demo as video")
        parser.add_argument("--video-fps", type=int, default=16,
                            help="video fps")
        parser.add_argument("--bounding-box-type", type=str, default='relevant',
                            help="options: relevant vs irrelevant")
        # human in the loop config
        parser.add_argument("--feedback-freq", type=int, default=4,
                            help="the frequency to collect human feedback")
        parser.add_argument("--no-feedback-after", type=int, default=10000000,
                            help="no feedback after the specified episode")
        parser.add_argument("--feedback-freq-decay", type=float, default=0,
                            help="gradually decrease the frequency to query human")
        # human interface config
        parser.add_argument("--host", type=str, default="127.0.0.1:5000",
                            help="the url to connect to the server/host.")
        parser.add_argument("--local-host", type=str, default="127.0.0.1:5000",
                            help="the url to for the local agent to communicate with the human interface.")
        parser.add_argument("--protocol", type=str, default="ws://",
                            help="The connection protocol (Default: ws://).")
        parser.add_argument("--server-type", type=str, default="sanic",
                            help="Options: fastapi (outdated), sanic")
        parser.add_argument("--server-debug", action="store_true", default=False,
                            help="whether to run the server in debug mode.")

        # save system args in agent config
        args, unknown = parser.parse_known_args()
        for arg in vars(args):
            self.agent_config.sys_args[arg] = getattr(args, arg)
        if self.agent_config.sys_args['human']:
            self.agent_config.sys_args['n_episodes'] = 100

        return parser

    def get_agent_config(self):
        super(Default_Human_Config, self).get_agent_config()
