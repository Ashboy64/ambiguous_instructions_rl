"""
Visualize start state, instruction, and ambiguous version for a bunch 
of different episodes for the specified environment.
"""


import gym 
import babyai
import time
from ambiguity import make_ambiguous
from babyai.levels.levelgen import *


# ENV_NAME = "BabyAI-GoToLocal-v0"
ENV_NAME = "BabyAI-GoTo-v0"


def main():
    print(level_dict)

    env = gym.make(ENV_NAME)
    env.render("human")

    for i in range(10):
        obs = env.reset()
        env.render()
        
        print()
        print(obs['mission'])
        print(make_ambiguous(env.instrs, env))

        start = time.time()

        while time.time() - start < 5:
            pass 


if __name__ == '__main__':
    main()