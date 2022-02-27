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
from tqdm import tqdm, trange
import copy

import babyai.utils as utils

from ambiguity import make_ambiguous

"""
python3 -u collect_questions_data.py \
--savedir ./test_data \
--max-steps 300 \
--eps-per-level 100 \
--samples-per-level 200


Specify which levels to collect data for by editing the LEVELS constant
"""


LEVELS = ["BabyAI-GoToObj-v0",
          "BabyAI-GoToRedBallGrey-v0",
          "BabyAI-GoToRedBall-v0",
          "BabyAI-GoToLocal-v0",
          # "BabyAI-PutNextLocal-v0",
          # "BabyAI-PickupLoc-v0",
          "BabyAI-GoToObjMaze-v0"]

def generate_data(levels, savedir, max_steps, eps_per_level, samples_per_level):
    assert max_steps > 0
    assert eps_per_level > 0
    assert samples_per_level > 0

    for level in levels:
        env = gym.make(level)
        agent = utils.load_agent(env, "BOT", argmax=False)
        demos_path = os.path.join(savedir, level + ".pkl")
        all_data = []

        for ep_idx in trange(eps_per_level, desc=f"Generating data for {level}..."):
            obs = env.reset()
            agent.on_reset()

            mission = obs["mission"]

            done = False
            t = 0
            while not done:
                action = agent.act(obs)['action']
                if isinstance(action, torch.Tensor):
                    action = action.item()
                new_obs, reward, done, _ = env.step(action)
                agent.analyze_feedback(reward, done)


                # This makes the instruction ambiguous that is being saved in the dataset,
                # but the original, unaltered instruction is still used by the agent
                is_ambiguous = np.random.uniform() > 0.5
                if is_ambiguous:
                    new_mission = make_ambiguous(env.instrs, env)
                    # print("\nmission:", mission)
                    # print("new_mission:", new_mission)
                else:
                    new_mission = mission

                all_data.append((blosc.pack_array(obs['image']), obs['direction'], reward, new_mission, is_ambiguous))
                obs = new_obs

                t+=1
                if t >= max_steps:
                    done = True

        sample_idxs = np.random.choice(list(range(len(all_data))), size=samples_per_level, replace=False)
        samples = [all_data[idx] for idx in sample_idxs]

        # if create_fake_positives:
        #     new_samples = []
        #     for sample in samples:
        #         if np.random.uniform() > 0.5:
        #             image, direction, reward, mission, is_ambiguous = sample
        #             words = mission.split(" ")
        #             drop_idx = np.random.choice(list(range(len(words))), size=1).item()
        #             new_words = [words[i] for i in range(len(words)) if i != drop_idx]
        #             ambiguous_mission = " ".join(new_words)
        #             new_sample = (image, direction, reward, ambiguous_mission, 1)
        #             new_samples.append(new_sample)
        #         else:
        #             new_samples.append(sample)
        #     samples = new_samples

        print("Saving samples...")
        utils.save_demos(samples, demos_path)
        print("{} samples saved to \"{}\"".format(len(samples), demos_path))



def parse_arguments():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--savedir', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--eps-per-level', type=int, default=100)
    parser.add_argument('--samples-per-level', type=int, default=200)
    # parser.add_argument('--create-fake-positives', action="store_true", default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    generate_data(LEVELS, args.savedir, args.max_steps, args.eps_per_level, args.samples_per_level)
