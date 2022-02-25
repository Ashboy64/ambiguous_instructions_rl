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


class AmbiguousInstructionsWrapper:
    def __init__(self, env, prob_ambiguous=0.5):
        self._env = env
        self._prob_ambiguous = prob_ambiguous
        self._true_mission = self._env.mission
        self._mission = self._true_mission
        self._is_currently_ambiguous = False

    def reset(self, *args, **kwargs):
        obs = self._env.reset(*args, **kwargs)
        self._true_mission = self._env.mission

        if np.random.uniform() < self._prob_ambiguous:
            self._is_currently_ambiguous = True
            self._mission = make_ambiguous(self._env.instrs, self._env)
        else:
            self._is_currently_ambiguous = False
            self._mission = self._true_mission

        obs["mission"] = self.mission
        return obs

    def step(self, *args, **kwargs):
        obs, reward, done, info = self._env.step(*args, **kwargs)
        obs["mission"] = self.mission
        return obs, reward, done, info

    @property
    def mission(self):
        return self._mission

    @property
    def true_mission(self):
        return self._true_mission

    @property
    def is_currently_ambiguous(self):
        return self._is_currently_ambiguous

    def __getattr__(self, name):
        return getattr(self._env, name)



if __name__ == "__main__":
    env = gym.make("BabyAI-GoToRedBall-v0")
    env = AmbiguousInstructionsWrapper(env, prob_ambiguous=0.5)

    for i in range(20):
        env.reset()
        print("\nenv.is_currently_ambiguous:", env.is_currently_ambiguous)
        print("env.true_mission:", env.true_mission)
        print("env.mission:", env.mission)

        for j in range(5):
            obs, reward, done, info = env.step(env.action_space.sample())
            print("\tobs[\"mission\"]:", obs["mission"])
