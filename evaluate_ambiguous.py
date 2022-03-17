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
from lstm import LSTM_classifier
from transformers import GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification
# Parse arguments

from torchtext.data.utils import get_tokenizer
import torch


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
parser.add_argument("--classifier-network", type=str, choices=["none", "gpt2", "lstm"], default="none",
                    help="Whether to use no classifier network, gpt2, or the lstm model")
parser.add_argument("--classifier-network-weights", type=str, default=None,
                    help="Path to the saved weights for the classifier network")
parser.add_argument("--device", type=str, default="cuda",
                    help="Path to the saved weights for the classifier network")

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

"""
# F 35296 | FPS 337 | D 0:01:44 | R:xsmM 0.536 0.399 0.000 0.950 | S 0.672 | F:xsmM 34.5 28.4 4 72
python3 -u evaluate_ambiguous.py \
--env BabyAI-PutNextLocalS6N4-v0 \
--episodes 1000 \
--model BabyAI-PutNextLocalS6N4-v0_plain_IL_bow_endpool_res_gru_seed1_22-03-09-23-51-59_best

# F 45068 | FPS 375 | D 0:02:00 | R:xsmM 0.401 0.412 0.000 0.950 | S 0.514 | F:xsmM 44.0 29.2 4 72
python3 -u evaluate_ambiguous.py \
--env BabyAI-PutNextLocalS6N4-v0 \
--episodes 1000 \
--ambiguous \
--prob-ambiguous 0.5 \
--model BabyAI-PutNextLocalS6N4-v0_ambiguous-0.5_IL_bow_endpool_res_gru_seed1_22-03-09-23-53-48_best

F 44119 | FPS 40 | D 0:18:23 | R:xsmM 0.415 0.413 0.000 0.950 | S 0.532 | F:xsmM 43.1 29.3 4 72
python3 -u evaluate_ambiguous.py \
--env BabyAI-PutNextLocalS6N4-v0 \
--episodes 1000 \
--ambiguous \
--prob-ambiguous 0.5 \
--model BabyAI-PutNextLocalS6N4-v0_ambiguous-0.5_IL_bow_endpool_res_gru_seed1_22-03-09-23-53-48_best \
--classifier-network lstm \
--classifier-network-weights classifier_models/lstm.pth \
--device cuda:1

F 43250 | FPS 18 | D 0:39:47 | R:xsmM 0.427 0.412 0.000 0.950 | S 0.550 | F:xsmM 42.2 29.3 4 72
python3 -u evaluate_ambiguous.py \
--env BabyAI-PutNextLocalS6N4-v0 \
--episodes 1000 \
--ambiguous \
--prob-ambiguous 0.5 \
--model BabyAI-PutNextLocalS6N4-v0_ambiguous-0.5_IL_bow_endpool_res_gru_seed1_22-03-09-23-53-48_best \
--classifier-network gpt2 \
--classifier-network-weights classifier_models/gpt2.pth \
--device cuda:1

# F 56093 | FPS 456 | D 0:02:02 | R:xsmM 0.249 0.370 0.000 0.950 | S 0.334 | F:xsmM 54.8 26.0 4 72
python3 -u evaluate_ambiguous.py \
--env BabyAI-PutNextLocalS6N4-v0 \
--episodes 1000 \
--nonsense \
--model BabyAI-PutNextLocalS6N4-v0_nonsense_IL_bow_endpool_res_gru_seed1_22-03-09-23-54-28_best



# F 10924 | FPS 292 | D 0:00:37 | R:xsmM 0.846 0.212 0.000 0.986 | S 0.959 | F:xsmM 10.7 13.9 1 64
python3 -u evaluate_ambiguous.py \
--env BabyAI-GoToLocal-v0 \
--episodes 1000 \
--model BabyAI-GoToLocal-v0_plain_IL_bow_endpool_res_gru_seed1_22-03-11-10-52-54_best

# F 16568 | FPS 245 | D 0:01:07 | R:xsmM 0.760 0.307 0.000 0.986 | S 0.880 | F:xsmM 16.2 19.8 1 64
python3 -u evaluate_ambiguous.py \
--env BabyAI-GoToLocal-v0 \
--episodes 1000 \
--ambiguous \
--prob-ambiguous 0.5 \
--model BabyAI-GoToLocal-v0_ambiguous-0.5_IL_bow_endpool_res_gru_seed1_22-03-11-10-53-02_best

F 15253 | FPS 11 | D 0:23:59 | R:xsmM 0.780 0.291 0.000 0.986 | S 0.897 | F:xsmM 14.9 18.8 1 64
python3 -u evaluate_ambiguous.py \
--env BabyAI-GoToLocal-v0 \
--episodes 1000 \
--ambiguous \
--prob-ambiguous 0.5 \
--model BabyAI-GoToLocal-v0_ambiguous-0.5_IL_bow_endpool_res_gru_seed1_22-03-11-10-53-02_best \
--classifier-network lstm \
--classifier-network-weights classifier_models/lstm.pth \
--device cuda:1

F 14930 | FPS 7 | D 0:36:58 | R:xsmM 0.785 0.287 0.000 0.986 | S 0.900 | F:xsmM 14.6 18.5 1 64
python3 -u evaluate_ambiguous.py \
--env BabyAI-GoToLocal-v0 \
--episodes 1000 \
--ambiguous \
--prob-ambiguous 0.5 \
--model BabyAI-GoToLocal-v0_ambiguous-0.5_IL_bow_endpool_res_gru_seed1_22-03-11-10-53-02_best \
--classifier-network gpt2 \
--classifier-network-weights classifier_models/gpt2.pth \
--device cuda:1


# F 29676 | FPS 359 | D 0:01:22 | R:xsmM 0.563 0.393 0.000 0.986 | S 0.706 | F:xsmM 29.0 25.0 1 64
python3 -u evaluate_ambiguous.py \
--env BabyAI-GoToLocal-v0 \
--episodes 1000 \
--nonsense \
--model BabyAI-GoToLocal-v0_nonsense_IL_bow_endpool_res_gru_seed1_22-03-11-10-53-22_best


# F 28931 | FPS 278 | D 0:01:44 | R:xsmM 0.565 0.441 0.000 0.972 | S 0.627 | F:xsmM 28.3 28.0 2 64
python3 -u evaluate_ambiguous.py \
--env BabyAI-PickupLoc-v0 \
--episodes 1000 \
--model BabyAI-PickupLoc-v0_plain_IL_bow_endpool_res_gru_seed1_22-03-11-10-53-59_best

# F 40823 | FPS 316 | D 0:02:09 | R:xsmM 0.381 0.452 0.000 0.972 | S 0.421 | F:xsmM 39.9 28.6 2 64
python3 -u evaluate_ambiguous.py \
--env BabyAI-PickupLoc-v0 \
--episodes 1000 \
--ambiguous \
--prob-ambiguous 0.5 \
--model BabyAI-PickupLoc-v0_ambiguous-0.5_IL_bow_endpool_res_gru_seed1_22-03-11-10-54-07_best

F 35474 | FPS 24 | D 0:24:34 | R:xsmM 0.464 0.455 0.000 0.972 | S 0.516 | F:xsmM 34.6 28.9 2 64
python3 -u evaluate_ambiguous.py \
--env BabyAI-PickupLoc-v0 \
--episodes 1000 \
--ambiguous \
--prob-ambiguous 0.5 \
--model BabyAI-PickupLoc-v0_ambiguous-0.5_IL_bow_endpool_res_gru_seed1_22-03-11-10-54-07_best \
--classifier-network lstm \
--classifier-network-weights classifier_models/lstm.pth \
--device cuda:1

F 35607 | FPS 16 | D 0:36:59 | R:xsmM 0.462 0.455 0.000 0.972 | S 0.515 | F:xsmM 34.8 28.9 2 64
python3 -u evaluate_ambiguous.py \
--env BabyAI-PickupLoc-v0 \
--episodes 1000 \
--ambiguous \
--prob-ambiguous 0.5 \
--model BabyAI-PickupLoc-v0_ambiguous-0.5_IL_bow_endpool_res_gru_seed1_22-03-11-10-54-07_best \
--classifier-network gpt2 \
--classifier-network-weights classifier_models/gpt2.pth \
--device cuda:1


# F 54135 | FPS 368 | D 0:02:27 | R:xsmM 0.176 0.359 0.000 0.972 | S 0.198 | F:xsmM 52.9 22.7 2 64
python3 -u evaluate_ambiguous.py \
--env BabyAI-PickupLoc-v0 \
--episodes 1000 \
--nonsense \
--model BabyAI-PickupLoc-v0_nonsense_IL_bow_endpool_res_gru_seed1_22-03-11-10-54-27_best





# F 18149 | FPS 134 | D 0:02:15 | R:xsmM 0.971 0.129 0.000 0.997 | S 0.983 | F:xsmM 17.7 74.5 2 576
python3 -u evaluate_ambiguous.py \
--env BabyAI-OpenDoorLoc-v0 \
--episodes 1000 \
--model BabyAI-OpenDoorLoc-v0_plain_IL_bow_endpool_res_gru_seed1_22-03-14-01-00-15_best

# F 34140 | FPS 172 | D 0:03:18 | R:xsmM 0.947 0.133 0.000 0.997 | S 0.992 | F:xsmM 33.3 81.5 2 576
python3 -u evaluate_ambiguous.py \
--env BabyAI-OpenDoorLoc-v0 \
--episodes 1000 \
--ambiguous \
--prob-ambiguous 0.5 \
--model BabyAI-OpenDoorLoc-v0_ambiguous-0.5_IL_bow_endpool_res_gru_seed1_22-03-14-01-01-48_best

# F 25915 | FPS 6 | D 1:15:25 | R:xsmM 0.960 0.113 0.000 0.997 | S 0.995 | F:xsmM 25.3 69.5 2 576
python3 -u evaluate_ambiguous.py \
--env BabyAI-OpenDoorLoc-v0 \
--episodes 1000 \
--ambiguous \
--prob-ambiguous 0.5 \
--model BabyAI-OpenDoorLoc-v0_ambiguous-0.5_IL_bow_endpool_res_gru_seed1_22-03-14-01-01-48_best \
--classifier-network lstm \
--classifier-network-weights classifier_models/lstm.pth \
--device cuda:1

F 25915 | FPS 4 | D 1:59:13 | R:xsmM 0.960 0.113 0.000 0.997 | S 0.995 | F:xsmM 25.3 69.5 2 576
python3 -u evaluate_ambiguous.py \
--env BabyAI-OpenDoorLoc-v0 \
--episodes 1000 \
--ambiguous \
--prob-ambiguous 0.5 \
--model BabyAI-OpenDoorLoc-v0_ambiguous-0.5_IL_bow_endpool_res_gru_seed1_22-03-14-01-01-48_best \
--classifier-network gpt2 \
--classifier-network-weights classifier_models/gpt2.pth \
--device cuda:1

# F 45851 | FPS 224 | D 0:03:24 | R:xsmM 0.930 0.105 0.000 0.997 | S 0.999 | F:xsmM 44.8 66.6 2 576
python3 -u evaluate_ambiguous.py \
--env BabyAI-OpenDoorLoc-v0 \
--episodes 1000 \
--nonsense \
--model BabyAI-OpenDoorLoc-v0_nonsense_IL_bow_endpool_res_gru_seed1_22-03-14-01-02-13_best
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


    if args.classifier_network == "gpt2":
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt2', num_labels=2)
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2')
        classifier_network = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='gpt2', config=model_config)
        classifier_network.resize_token_embeddings(len(tokenizer))
        classifier_network.config.pad_token_id = classifier_network.config.eos_token_id
        classifier_network.load_state_dict(torch.load(args.classifier_network_weights))
        classifier_network.eval()
        classifier_network = classifier_network.to(args.device)
        vocab = None
    elif args.classifier_network == "lstm":
        from lstm import create_vocab
        vocab = create_vocab(['lstm_vocab/train.jsonl', 'lstm_vocab/valid.jsonl'])
        classifier_network = LSTM_classifier(len(vocab))
        classifier_network.load_state_dict(torch.load(args.classifier_network_weights))
        classifier_network.eval()
        classifier_network = classifier_network.to(args.device)
        tokenizer = get_tokenizer("spacy", language='en_core_web_sm')
    elif args.classifier_network == "none":
        vocab = None
        classifier_network = None
        tokenizer = None
    else:
        raise ValueError("Can't use args.classifier_network = \"{args.classifier_network}\".")

    # Evaluate
    if isinstance(agent, utils.DemoAgent):
        logs = evaluate_demo_agent(agent, episodes)
    elif isinstance(agent, utils.BotAgent) or args.contiguous_episodes:
        logs = evaluate(agent, env, episodes, False)
    else:
        # logs = batch_evaluate(agent, args.env, seed, episodes)
        logs = batch_evaluate(agent,
                              args.env,
                              seed,
                              episodes,
                              nonsense=args.nonsense,
                              ambiguous=args.ambiguous,
                              prob_ambiguous=args.prob_ambiguous,
                              classifier_network=classifier_network,
                              vocab=vocab,
                              tokenizer=tokenizer)


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


        rate_ambiguous_per_episode = utils.synthesize(logs["rate_ambiguous_per_episode"])
        print("rate_ambiguous_per_episode {:.3f}".format(rate_ambiguous_per_episode["mean"]))

        rate_unambiguous_per_episode = utils.synthesize(logs["rate_unambiguous_per_episode"])
        print("rate_unambiguous_per_episode {:.3f}".format(rate_unambiguous_per_episode["mean"]))

        rate_correct_pred_per_episode = utils.synthesize(logs["rate_correct_pred_per_episode"])
        print("rate_correct_pred_per_episode {:.3f}".format(rate_correct_pred_per_episode["mean"]))

        rate_tp_per_episode = utils.synthesize(logs["rate_tp_per_episode"])
        print("rate_tp_per_episode {:.3f}".format(rate_tp_per_episode["mean"]))

        rate_tn_episode = utils.synthesize(logs["rate_tn_episode"])
        print("rate_tn_episode {:.3f}".format(rate_tn_episode["mean"]))

        rate_fp_per_episode = utils.synthesize(logs["rate_fp_per_episode"])
        print("rate_fp_per_episode {:.3f}".format(rate_fp_per_episode["mean"]))

        rate_fn_per_episode = utils.synthesize(logs["rate_fn_per_episode"])
        print("rate_fn_per_episode {:.3f}".format(rate_fn_per_episode["mean"]))

        n_ambiguous_per_episode = utils.synthesize(logs["n_ambiguous_per_episode"])
        print("n_ambiguous_per_episode {:.3f}".format(n_ambiguous_per_episode["mean"]))

        n_unambiguous_per_episode = utils.synthesize(logs["n_unambiguous_per_episode"])
        print("n_unambiguous_per_episode {:.3f}".format(n_unambiguous_per_episode["mean"]))

        n_correct_pred_per_episode = utils.synthesize(logs["n_correct_pred_per_episode"])
        print("n_correct_pred_per_episode {:.3f}".format(n_correct_pred_per_episode["mean"]))

        n_tp_per_episode = utils.synthesize(logs["n_tp_per_episode"])
        print("n_tp_per_episode {:.3f}".format(n_tp_per_episode["mean"]))

        n_tn_episode = utils.synthesize(logs["n_tn_episode"])
        print("n_tn_episode {:.3f}".format(n_tn_episode["mean"]))

        n_fp_per_episode = utils.synthesize(logs["n_fp_per_episode"])
        print("n_fp_per_episode {:.3f}".format(n_fp_per_episode["mean"]))

        n_fn_per_episode = utils.synthesize(logs["n_fn_per_episode"])
        print("n_fn_per_episode {:.3f}".format(n_fn_per_episode["mean"]))

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
