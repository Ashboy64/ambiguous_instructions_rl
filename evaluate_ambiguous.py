#!/usr/bin/env python3

"""
Evaluate a trained model or bot
"""

import argparse
import gym
import time
import datetime

import babyai.utils as utils
from babyai.evaluate import evaluate_demo_agent, evaluate#, batch_evaluate,

from custom_evaluate import batch_evaluate
# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin or --demos REQUIRED)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model or --demos REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="name of the demos file (REQUIRED or --demos-origin or --model REQUIRED)")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes of evaluation (default: 1000)")
parser.add_argument("--seed", type=int, default=int(1e9),
                    help="random seed")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--contiguous-episodes", action="store_true", default=False,
                    help="Make sure episodes on which evaluation is done are contiguous")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="The number of worse episodes to show")

parser.add_argument("--nonsense", "-n", action="store_true", default=False,
                    help="Add NonsenseInstructionsWrapper")
parser.add_argument("--ambiguous", "-a", action="store_true", default=False,
                    help="Add AmbiguousInstructionsWrapper")
parser.add_argument("--prob-ambiguous", type=float, default=0.5,
                    help="Probability that the env will generate ambiguous instructions each episode")


"""
# R:xsmM 0.534 0.407 0.000 0.950 | S 0.657
python3 -u evaluate_ambiguous.py \
--env BabyAI-PutNextLocalS6N4-v0 \
--episodes 1000 \
--model BabyAI-PutNextLocalS6N4-v0_IL_bow_endpool_res_gru_seed1_22-03-07-20-35-54_best

# babyai
# R:xsmM 0.269 0.380 0.000 0.938 | S 0.362
python3 -u evaluate_ambiguous.py \
--env BabyAI-PutNextLocalS6N4-v0 \
--episodes 1000 \
--model BabyAI-PutNextLocalS6N4-v0_IL_bow_endpool_res_gru_seed1_22-03-08-10-01-33_best \
--ambiguous \
--prob-ambiguous 1

# babyai2
# R:xsmM 0.220 0.357 0.000 0.950 | S 0.301
python3 -u evaluate_ambiguous.py \
--env BabyAI-PutNextLocalS6N4-v0 \
--episodes 1000 \
--model BabyAI-PutNextLocalS6N4-v0_IL_bow_endpool_res_gru_seed1_22-03-08-10-01-41_best \
--nonsense


# babyai3
# R:xsmM 0.960 0.149 0.000 0.997 | S 0.982
python3 -u evaluate_ambiguous.py \
--env BabyAI-OpenDoorLoc-v0 \
--episodes 1000 \
--model BabyAI-OpenDoorLoc-v0_IL_bow_endpool_res_gru_seed1_22-03-08-09-52-57_best

# babyai4
# R:xsmM 0.907 0.166 0.000 0.997 | S 0.989
python3 -u evaluate_ambiguous.py \
--env BabyAI-OpenDoorLoc-v0 \
--episodes 1000 \
--model BabyAI-OpenDoorLoc-v0_IL_bow_endpool_res_gru_seed1_22-03-08-09-54-23_best \
--ambiguous \
--prob-ambiguous 1

# babyai5
# R:xsmM 0.930 0.105 0.000 0.997 | S 0.999
python3 -u evaluate_ambiguous.py \
--env BabyAI-OpenDoorLoc-v0 \
--episodes 1000 \
--model BabyAI-OpenDoorLoc-v0_IL_bow_endpool_res_gru_seed1_22-03-08-09-57-30_best \
--nonsense


# babyai6
# R:xsmM 0.840 0.220 0.000 0.986 | S 0.955
python3 -u evaluate_ambiguous.py \
--env BabyAI-GoToLocal-v0 \
--episodes 1000 \
--model BabyAI-GoToLocal-v0_IL_bow_endpool_res_gru_seed1_22-03-08-09-58-26_best

# babyai7
# R:xsmM 0.729 0.309 0.000 0.986 | S 0.884
python3 -u evaluate_ambiguous.py \
--env BabyAI-GoToLocal-v0 \
--episodes 1000 \
--model BabyAI-GoToLocal-v0_IL_bow_endpool_res_gru_seed1_22-03-08-10-00-27_best \
--ambiguous \
--prob-ambiguous 1

# babyai8
# R:xsmM 0.605 0.354 0.000 0.986 | S 0.817
python3 -u evaluate_ambiguous.py \
--env BabyAI-GoToLocal-v0 \
--episodes 1000 \
--model BabyAI-GoToLocal-v0_IL_bow_endpool_res_gru_seed1_22-03-08-10-00-32_best \
--nonsense


python3 -u evaluate_ambiguous.py \
--env BabyAI-PickupLoc-v0 \
--episodes 1000 \
--model BabyAI-PickupLoc-v0_IL_bow_endpool_res_gru_seed1_22-03-08-10-31-21_best \
--ambiguous \
--prob-ambiguous 1

"""

def main(args, seed, episodes):
    # Set seed for all randomness sources
    utils.seed(seed)

    # Define agent

    env = gym.make(args.env)
    env.seed(seed)
    agent = utils.load_agent(env, args.model, args.demos, args.demos_origin, args.argmax, args.env)
    if args.model is None and args.episodes > len(agent.demos):
        # Set the number of episodes to be the number of demos
        episodes = len(agent.demos)

    # Evaluate
    if isinstance(agent, utils.DemoAgent):
        logs = evaluate_demo_agent(agent, episodes)
    elif isinstance(agent, utils.BotAgent) or args.contiguous_episodes:
        logs = evaluate(agent, env, episodes, False)
    else:
        # logs = batch_evaluate(agent, args.env, seed, episodes)
        logs = batch_evaluate(agent, args.env, seed, episodes, nonsense=args.nonsense, ambiguous=args.ambiguous, prob_ambiguous=args.prob_ambiguous)


    return logs


if __name__ == "__main__":
    args = parser.parse_args()
    assert_text = "ONE of --model or --demos-origin or --demos must be specified."
    assert int(args.model is None) + int(args.demos_origin is None) + int(args.demos is None) == 2, assert_text

    start_time = time.time()
    logs = main(args, args.seed, args.episodes)
    end_time = time.time()

    # Print logs
    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames/(end_time - start_time)
    ellapsed_time = int(end_time - start_time)
    duration = datetime.timedelta(seconds=ellapsed_time)

    if args.model is not None:
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        success_per_episode = utils.synthesize(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])

    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    if args.model is not None:
        print("F {} | FPS {:.0f} | D {} | R:xsmM {:.3f} {:.3f} {:.3f} {:.3f} | S {:.3f} | F:xsmM {:.1f} {:.1f} {} {}"
              .format(num_frames, fps, duration,
                      *return_per_episode.values(),
                      success_per_episode['mean'],
                      *num_frames_per_episode.values()))
    else:
        print("F {} | FPS {:.0f} | D {} | F:xsmM {:.1f} {:.1f} {} {}"
              .format(num_frames, fps, duration, *num_frames_per_episode.values()))

    indexes = sorted(range(len(logs["num_frames_per_episode"])), key=lambda k: - logs["num_frames_per_episode"][k])

    n = args.worst_episodes_to_show
    if n > 0:
        print("{} worst episodes:".format(n))
        for i in indexes[:n]:
            if 'seed_per_episode' in logs:
                print(logs['seed_per_episode'][i])
            if args.model is not None:
                print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
            else:
                print("- episode {}: F={}".format(i, logs["num_frames_per_episode"][i]))
