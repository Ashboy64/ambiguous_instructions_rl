#!/usr/bin/env python3

"""
Script to train agent through imitation learning using demonstrations.
"""

import os
import csv
import copy
import gym
import time
import datetime
import numpy as np
import sys
import logging
import torch
from babyai.arguments import ArgumentParser
import babyai.utils as utils
from imitation_ambiguous import ImitationLearningAmbiguous

from utils import save_args

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or demos-origin or multi-demos required)")
parser.add_argument("--demos-origin", required=False,
                    help="origin of the demonstrations: human | agent (REQUIRED or demos or multi-demos required)")
parser.add_argument("--episodes", type=int, default=0,
                    help="number of episodes of demonstrations to use"
                         "(default: 0, meaning all demos)")
parser.add_argument("--multi-env", nargs='*', default=None,
                  help="name of the environments used for validation/model loading")
parser.add_argument("--multi-demos", nargs='*', default=None,
                    help="demos filenames for envs to train on (REQUIRED when multi-env is specified)")
parser.add_argument("--multi-episodes", type=int, nargs='*', default=None,
                    help="number of episodes of demos to use from each file (REQUIRED when multi-env is specified)")
parser.add_argument("--save-interval", type=int, default=1,
                    help="number of epochs between two saves (default: 1, 0 means no saving)")

parser.add_argument("--nonsense", "-n", action="store_true", default=False,
                    help="Add NonsenseInstructionsWrapper")
parser.add_argument("--ambiguous", "-a", action="store_true", default=False,
                    help="Add AmbiguousInstructionsWrapper")
parser.add_argument("--prob-ambiguous", type=float, default=0.5,
                    help="Probability that the env will generate ambiguous instructions each episode")

"""
OpenDoorLoc
GoToLocal
PickupLoc
PutNextLocalS6N4

GoToObjMaze

python3 -u train_il_eval_ambiguous.py \
--tb \
--patience 1000 \
--demos BabyAI-PickupLoc-v0 \
--env BabyAI-PickupLoc-v0

python3 -u train_il_eval_ambiguous.py \
--tb \
--patience 1000 \
--demos BabyAI-PickupLoc-v0_ambiguous \
--env BabyAI-PickupLoc-v0 \
--ambiguous \
--prob-ambiguous 1

python3 -u train_il_eval_ambiguous.py \
--tb \
--patience 1000 \
--demos BabyAI-PickupLoc-v0_nonsense \
--env BabyAI-PickupLoc-v0 \
--nonsense
"""


def main(args):
    # Verify the arguments when we train on multiple environments
    # No need to check for the length of len(args.multi_env) in case, for some reason, we need to validate on other envs
    if args.multi_env is not None:
        assert len(args.multi_demos) == len(args.multi_episodes)


    if args.pretrained_model:
        default_model_name = args.pretrained_model + '_pretrained_' + ImitationLearning.default_model_name(args)
    args.model = args.model.format(**model_name_parts) if args.model else ImitationLearningAmbiguous.default_model_name(args)

    utils.configure_logging(args.model)
    logger = logging.getLogger(__name__)

    save_args(args, utils.get_log_dir(args.model))


    il_learn = ImitationLearningAmbiguous(args)

    # Define logger and Tensorboard writer
    header = (["update", "frames", "FPS", "duration", "entropy", "policy_loss", "train_accuracy"]
              + ["validation_accuracy"])
    if args.multi_env is None:
        header.extend(["validation_return", "validation_success_rate"])
    else:
        header.extend(["validation_return_{}".format(env) for env in args.multi_env])
        header.extend(["validation_success_rate_{}".format(env) for env in args.multi_env])
    writer = None
    if args.tb:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(utils.get_log_dir(args.model))

    # Define csv writer
    csv_writer = None
    csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
    first_created = not os.path.exists(csv_path)
    # we don't buffer data going in the csv log, cause we assume
    # that one update will take much longer that one write to the log
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    if first_created:
        csv_writer.writerow(header)

    # Get the status path
    status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')

    # Log command, availability of CUDA, and model
    logger.info(args)
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))
    logger.info(il_learn.acmodel)

    il_learn.train(il_learn.train_demos, writer, csv_writer, status_path, header)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
