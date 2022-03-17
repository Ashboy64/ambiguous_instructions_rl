"""
Visualize start state, instruction, and ambiguous version for a bunch
of different episodes for the specified environment.

Environments that seem to contain decent amount of ambiguity:
- GoToLocal
- GoToSeqS5R2 (?)
- MiniBossLevel (?)
- PickupLoc
- PutNextLocal (maybe S6N4 version)
- Maybe SynthS5R2
- ActionObjDoor
- GoToObjDoor
- OpenDoorLoc

Final List:
- GoToLocal
- PickupLoc
- GoToObjDoor (a lot like GoToLocal, except more rooms as distractors?)
- Can try PutNextLocalS6N4 (like PutNextLocal but smaller grid so maybe easier to learn, but not as ambiguous)
- OpenDoorLoc


"""


import gym
import babyai
import time
import argparse
from ambiguity import make_ambiguous
from babyai.levels.levelgen import *


parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
args = parser.parse_args()


def main():
    # print(level_dict)

    env = gym.make(f"BabyAI-{args.env}-v0")
    env.render("human")

    for i in range(10):
        obs = env.reset()
        env.render()

        print()
        print(obs['mission'])
        print(make_ambiguous(env.instrs, env))

        start = time.time()

        # while time.time() - start < 5:
        #     pass


if __name__ == '__main__':
    main()
