import numpy as np
import babyai.utils as utils
import blosc
import json


def get_image_command(filepath):
    print(f"Loading data from \"{filepath}\"...")
    samples = utils.load_demos(filepath)
    sample_list = []
    labels = []
    for sample in samples:
        image, direction, reward, mission, label = sample
        sample_list.append(make_sample(image, mission))
        labels.append(label)
    print("Collected", len(sample_list), "sample.")
    return sample_list, labels


def make_sample(image, command):
    image = list(blosc.unpack_array(image).flatten().astype(str))
    sample = ' '.join(image) + ' ' + command
    return sample


def make_data_file(samples, labels, path):
    for idx in range(len(samples)):
        dp = {}
        dp['text'] = samples[idx]
        dp['label'] = labels[idx]
        # j_data = json.dumps(dp)
        with open(path, 'a') as outfile:
            json.dump(dp, outfile)
            outfile.write("\n")


def main():
    # samples, labels = get_image_command("data/BabyAI-GoToLocal-v0.pkl")
    # make_data_file(samples, labels, "data/data.jsonl")

    samples, labels = get_image_command("data/BabyAI-GoToObj-v0.pkl")
    make_data_file(samples, labels, "data/data_val.jsonl")

if __name__ == '__main__':
    main()