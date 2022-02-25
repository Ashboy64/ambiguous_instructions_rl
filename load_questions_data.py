import os
import numpy as np
import blosc

import babyai.utils as utils

"""
python3 -u load_questions_data.py \
--filepath ./test_data/BabyAI-GoToLocal-v0.pkl
"""

def load_data(filepath):
    print(f"Loading data from \"{filepath}\"...")
    samples = utils.load_demos(filepath)
    for sample in samples:
        image, direction, reward, mission, label = sample
        image = blosc.unpack_array(image)
        print("\nimage.shape:", image.shape)
        print("direction:", direction)
        print("reward:", reward)
        print("mission:", mission)
        print("label:", label)
        import pdb; pdb.set_trace()

def parse_arguments():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--filepath', type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    load_data(args.filepath)
