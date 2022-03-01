#!/usr/bin/env python3

"""
Visualize the performance of a model on a given environment.
"""
import os
import numpy as np
import argparse
import gym
import time
import cv2

import babyai.utils as utils

from wrappers import AmbiguousInstructionsWrapper, NonsenseInstructionsWrapper

"""
python3 -u render_demos.py \
--env BabyAI-GoToLocal-v0 \
--demos /iris/u/khatch/CS224N/ambiguous_instructions_rl/test_data/BabyAI-GoToLocal-v0 \
--demos-origin agent \
--n-episodes 1

python3 -u render_demos.py \
--env BabyAI-PickupLoc-v0 \
--model BabyAI-PickupLoc-v0_nonsense_ppo_bow_endpool_res_gru_mem_seed1_22-02-26-22-07-48 \
--nonsense \
--n-episodes 5


python3 -u render_demos.py \
--env BabyAI-PutNextLocalS6N4-v0 \
--model BabyAI-PutNextLocalS6N4-v0_nonsense_ppo_bow_endpool_res_gru_mem_seed1_22-02-26-22-09-17 \
--nonsense \
--n-episodes 5



"""

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin or --demos REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or --model demos-origin required)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model or --demos REQUIRED)")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (default: 0 if model agent, 1 if demo agent)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--pause", type=float, default=0.1,
                    help="the pause between two consequent actions of an agent")
parser.add_argument("--manual-mode", action="store_true", default=False,
                    help="Allows you to take control of the agent at any point of time")

parser.add_argument("--n-episodes", type=int, default=1,
                    help="How many episodes from the demo file to render")
parser.add_argument("--nonsense", "-n", action="store_true", default=False,
                    help="Add NonsenseInstructionsWrapper")
parser.add_argument("--ambiguous", "-a", action="store_true", default=False,
                    help="Add AmbiguousInstructionsWrapper")
parser.add_argument("--prob-ambiguous", type=float, default=0.5,
                    help="Probability that the env will generate ambiguous instructions each episode")
args = parser.parse_args()

action_map = {
    "left"      : "left",
    "right"     : "right",
    "up"        : "forward",
    "p"         : "pickup",
    "pageup"    : "pickup",
    "d"         : "drop",
    "pagedown"  : "drop",
    " "         : "toggle"
}

assert args.model is not None or args.demos is not None, "--model or --demos must be specified."
if args.seed is None:
    args.seed = 0 if args.model is not None else 1

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)

if args.ambiguous:
    env = AmbiguousInstructionsWrapper(env, prob_ambiguous=args.prob_ambiguous)
elif args.nonsense:
    env = NonsenseInstructionsWrapper(env)

global obs
obs = env.reset()
print("Mission: {}".format(obs["mission"]))
if args.ambiguous or args.nonsense:
    print("True mission: {}".format(env.true_mission))

# Define agent
agent = utils.load_agent(env, args.model, args.demos, args.demos_origin, args.argmax, args.env)


def save_images(images, model_name, episode_num):
    episode_images_dir = os.path.join("rendered_demos", model_name, f"ep-{episode_num}")

    if not os.path.isdir(episode_images_dir):
        os.makedirs(episode_images_dir)

    for i, image in enumerate(images):
        image = np.flip(image, axis=-1)
        image_file = os.path.join(episode_images_dir, f"frame{i}.png")
        cv2.imwrite(image_file, image)


images = []
step = 0
episode_num = 1
while True:
    img = env.render("rgb_array")
    images.append(img)

    result = agent.act(obs)
    obs, reward, done, _ = env.step(result['action'])
    agent.analyze_feedback(reward, done)
    if 'dist' in result and 'value' in result:
        dist, value = result['dist'], result['value']
        dist_str = ", ".join("{:.4f}".format(float(p)) for p in dist.probs[0])
        # print("step: {}, mission: {}, dist: {}, entropy: {:.2f}, value: {:.2f}".format(
        #     step, obs["mission"], dist_str, float(dist.entropy()), float(value)))
    else:
        print("step: {}, mission: {}".format(step, obs['mission']))
    if done:
        # print("Reward:", reward)
        print("Episode: {}, Mission: {}, True Mission: {}, Return: {}".format(episode_num, obs['mission'], env.true_mission, reward))

        save_images(images, args.model, episode_num)
        images = []

        episode_num += 1
        env.seed(args.seed + episode_num)
        obs = env.reset()
        agent.on_reset()
        step = 0
        if episode_num > args.n_episodes:
            break
    else:
        step += 1
