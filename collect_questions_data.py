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
from babyai.levels.verifier import ObjDesc

from ambiguity import make_ambiguous
from gym_minigrid.minigrid import COLOR_NAMES

"""
python3 -u collect_questions_data.py \
--savedir ./data \
--max-steps 500 \
--eps-per-level 100 \
--samples-per-level 1000


Specify which levels to collect data for by editing the LEVELS constant
"""


LEVELS = [# "BabyAI-GoToObj-v0",
          # "BabyAI-GoToRedBallGrey-v0",
          # "BabyAI-GoToRedBall-v0",
          # "BabyAI-GoToLocal-v0",
          # "BabyAI-OpenDoorLoc-v0",
          "BabyAI-PutNextLocalS6N4-v0",
          # "BabyAI-PickupLoc-v0",
          # "BabyAI-GoToObjMaze-v0"
          ]

def generate_data(levels, savedir, max_steps, eps_per_level, samples_per_level):
    OBJ_TYPES = ['box', 'ball', 'key', 'door', 'object']
    LOC_NAMES = ['left', 'right', 'front', 'behind']

    assert max_steps > 0
    assert eps_per_level > 0
    assert samples_per_level > 0

    for level in levels:
        env = gym.make(level)
        agent = utils.load_agent(env, "BOT", argmax=False)
        demos_path = os.path.join(savedir, level + ".pkl")
        all_data = []
        total_ambi = 0

        for ep_idx in trange(eps_per_level, desc=f"Generating data for {level}..."):
            obs = env.reset()
            agent.on_reset()

            mission = obs["mission"]

            t = 0
            num_ambiguous = 0
            all = 0
            while True:
                # print("agent.act(obs)['action']", agent.act(obs)['action'])
                try:
                    action = agent.act(obs)['action']
                except:
                    print("Error in data generation.")
                if isinstance(action, torch.Tensor):
                    action = action.item()
                new_obs, reward, done, _ = env.step(action)
                agent.analyze_feedback(reward, done)

                # This makes the instruction ambiguous that is being saved in the dataset,
                # but the original, unaltered instruction is still used by the agent
                is_ambiguous = np.random.uniform() > 0.5
                if is_ambiguous:
                    new_mission = make_ambiguous(env.instrs, env)
                    # print(mission, new_mission)
                    mission_wordlist = new_mission.split(" ")
                    color = None
                    loc = None
                    type = None
                    descs = []
                    is_ambiguous = False
                    for idx in range(len(mission_wordlist)):
                        if mission_wordlist[idx] in OBJ_TYPES:
                            if mission_wordlist[idx] != "object":
                                type = mission_wordlist[idx]
                            if mission_wordlist[idx-1] in COLOR_NAMES:
                                color = mission_wordlist[idx-1]
                            for i in range(5):
                                if i + idx > len(mission_wordlist) - 1:
                                    break
                                else:
                                    if mission_wordlist[idx + i] in LOC_NAMES:
                                        loc = mission_wordlist[idx + i]
                            descs.append(ObjDesc(type, color=color, loc=loc))
                    # print(mission, new_mission)
                    # print("\nmission:", mission)
                    # print("new_mission:", color, loc, type)
                    for desc in descs:
                        if len(desc.find_matching_objs(env)[0]) > 1:
                            is_ambiguous = True
                else:
                    new_mission = mission
                if is_ambiguous:
                    state = blosc.pack_array(obs['image'])
                    present = 0
                    for data in all_data:
                        if state == data[0] and new_mission == data[-1]:
                            present = 1
                    if not present:
                        all_data.append((blosc.pack_array(obs['image']), obs['direction'], reward, new_mission, is_ambiguous))
                        num_ambiguous += 1
                        all += 1
                else:
                    state = blosc.pack_array(obs['image'])
                    present = 0
                    for data in all_data:
                        if state == data[0] and new_mission == data[-1]:
                            present = 1
                    if not present:
                        all_data.append((blosc.pack_array(obs['image']), obs['direction'], reward, new_mission, is_ambiguous))
                        all += 1
                obs = new_obs

                t+=1
                if t >= max_steps:
                    print("Generated", num_ambiguous, "ambiguous instructions in a total of", all, "instructions.")
                    break

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
