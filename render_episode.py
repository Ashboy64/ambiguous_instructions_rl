import gym
import time

import babyai.utils as utils


def render_episode():






def parse_arguments():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--filepath', type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    render_episode(args.filepath)
