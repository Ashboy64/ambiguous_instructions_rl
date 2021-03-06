#!/usr/bin/env python3

"""
Generate a set of agent demonstrations.

The agent can either be a trained model or the heuristic expert (bot).

Demonstration generation can take a long time, but it can be parallelized
if you have a cluster at your disposal. Provide a script that launches
make_agent_demos.py at your cluster as --job-script and the number of jobs as --jobs.


"""

import argparse
import gym
import logging
import sys
import subprocess
import os
import time
import numpy as np
import blosc
import torch

import babyai.utils as utils

from wrappers import NONSENSE_STRING
from ambiguity import make_ambiguous

# Parse arguments

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default='BOT',
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="path to save demonstrations (based on --model and --origin by default)")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes to generate demonstrations for")
parser.add_argument("--valid-episodes", type=int, default=512,
                    help="number of validation episodes to generate demonstrations for")
parser.add_argument("--seed", type=int, default=0,
                    help="start random seed")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--log-interval", type=int, default=100,
                    help="interval between progress reports")
parser.add_argument("--save-interval", type=int, default=10000,
                    help="interval between demonstrations saving")
parser.add_argument("--filter-steps", type=int, default=0,
                    help="filter out demos with number of steps more than filter-steps")
parser.add_argument("--on-exception", type=str, default='warn', choices=('warn', 'crash'),
                    help="How to handle exceptions during demo generation")

parser.add_argument("--job-script", type=str, default=None,
                    help="The script that launches make_agent_demos.py at a cluster.")
parser.add_argument("--jobs", type=int, default=0,
                    help="Split generation in that many jobs")

args = parser.parse_args()
logger = logging.getLogger(__name__)

# Set seed for all randomness sources

"""
python3 -u make_agent_demos_ambiguous.py \
--env BabyAI-OpenDoorLoc-v0 \
--model BOT \
--episodes 10000 \
--valid-episodes 1000 \
--demos BabyAI-OpenDoorLoc-v0
"""

def print_demo_lengths(demos):
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    logger.info('Demo length: {:.3f}+-{:.3f}'.format(
        np.mean(num_frames_per_episode), np.std(num_frames_per_episode)))


def generate_demos(n_episodes, valid, seed, shift=0):
    utils.seed(seed)

    # Generate environment
    env = gym.make(args.env)

    agent = utils.load_agent(env, args.model, args.demos, 'agent', args.argmax, args.env)
    demos_path = utils.get_demos_path(args.demos, args.env, 'agent', valid)

    if valid:
        ambiguous_demos_path = demos_path.split(".")[0][:-6] + "_ambiguous_valid.pkl"
        half_ambiguous_demos_path = demos_path.split(".")[0][:-6] + "_half_ambiguous_valid.pkl"
        nonsense_demos_path = demos_path.split(".")[0][:-6] + "_nonsense_valid.pkl"
    else:
        ambiguous_demos_path = demos_path.split(".")[0] + "_ambiguous.pkl"
        half_ambiguous_demos_path = demos_path.split(".")[0] + "_half_ambiguous.pkl"
        nonsense_demos_path = demos_path.split(".")[0] + "_nonsense.pkl"



    demos = []
    ambiguous_demos = []
    half_ambiguous_demos = []
    nonsense_demos = []

    checkpoint_time = time.time()

    just_crashed = False
    while True:
        if len(demos) == n_episodes:
            break

        done = False
        if just_crashed:
            logger.info("reset the environment to find a mission that the bot can solve")
            env.reset()
        else:
            env.seed(seed + len(demos))
        obs = env.reset()
        agent.on_reset()

        actions = []
        mission = obs["mission"]
        images = []
        directions = []

        try:
            while not done:
                action = agent.act(obs)['action']
                if isinstance(action, torch.Tensor):
                    action = action.item()
                new_obs, reward, done, _ = env.step(action)
                agent.analyze_feedback(reward, done)

                actions.append(action)
                images.append(obs['image'])
                directions.append(obs['direction'])

                obs = new_obs
            if reward > 0 and (args.filter_steps == 0 or len(images) <= args.filter_steps):
                demos.append((mission, blosc.pack_array(np.array(images)), directions, actions))

                ambiguous_mission = make_ambiguous(env.instrs, env)
                ambiguous_demos.append((ambiguous_mission, blosc.pack_array(np.array(images)), directions, actions))
                half_ambiguous_mission = ambiguous_mission if np.random.uniform() > 0.5 else mission
                half_ambiguous_demos.append((half_ambiguous_mission, blosc.pack_array(np.array(images)), directions, actions))
                nonsense_demos.append((NONSENSE_STRING, blosc.pack_array(np.array(images)), directions, actions))

                just_crashed = False

            if reward == 0:
                if args.on_exception == 'crash':
                    raise Exception("mission failed, the seed is {}".format(seed + len(demos)))
                just_crashed = True
                logger.info("mission failed")
        except (Exception, AssertionError):
            if args.on_exception == 'crash':
                raise
            just_crashed = True
            logger.exception("error while generating demo #{}".format(len(demos)))
            continue

        if len(demos) and len(demos) % args.log_interval == 0:
            now = time.time()
            demos_per_second = args.log_interval / (now - checkpoint_time)
            to_go = (n_episodes - len(demos)) / demos_per_second
            logger.info("demo #{}, {:.3f} demos per second, {:.3f} seconds to go".format(
                len(demos) - 1, demos_per_second, to_go))
            checkpoint_time = now

        # Save demonstrations

        if args.save_interval > 0 and len(demos) < n_episodes and len(demos) % args.save_interval == 0:
            logger.info("Saving demos...")
            utils.save_demos(demos, demos_path)
            utils.save_demos(half_ambiguous_demos, half_ambiguous_demos_path)
            utils.save_demos(ambiguous_demos, ambiguous_demos_path)
            utils.save_demos(nonsense_demos, nonsense_demos_path)
            logger.info("{} demos saved".format(len(demos)))
            # print statistics for the last 100 demonstrations
            print_demo_lengths(demos[-100:])


    # Save demonstrations
    logger.info("Saving demos...")

    # print("\n\nHere\n\n")
    # os.makedirs(demos_path)


    utils.save_demos(demos, demos_path)
    utils.save_demos(half_ambiguous_demos, half_ambiguous_demos_path)
    utils.save_demos(ambiguous_demos, ambiguous_demos_path)
    utils.save_demos(nonsense_demos, nonsense_demos_path)
    logger.info("{} demos saved".format(len(demos)))
    print_demo_lengths(demos[-100:])


logging.basicConfig(level='INFO', format="%(asctime)s: %(levelname)s: %(message)s")
logger.info(args)
# Training demos
if args.jobs == 0:
    generate_demos(args.episodes, False, args.seed)
else:
    raise NotImplementedError
    generate_demos_cluster()
# Validation demos
if args.valid_episodes:
    generate_demos(args.valid_episodes, True, int(1e9))
