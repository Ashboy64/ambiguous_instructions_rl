import numpy as np
import babyai.utils as utils
import blosc
import json
import os
import random


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


def split_dataset(data_dir):
    paths = os.listdir(data_dir)
    all_samples = []
    all_labels = []
    for path in paths:
        samples, labels = get_image_command(os.path.join(data_dir, path))
        all_samples += samples
        all_labels += labels
    length = len(all_samples)
    indices = list(range(length))
    random.shuffle(indices)
    train_samples = []
    train_labels = []
    for idx in indices[:int(0.7*length)]:
        train_samples.append(all_samples[idx])
        train_labels.append(all_labels[idx])
    make_data_file(train_samples, train_labels, "data/train.jsonl")
    valid_samples = []
    valid_labels = []
    for idx in indices[int(0.7*length):int(0.9*length)]:
        valid_samples.append(all_samples[idx])
        valid_labels.append(all_labels[idx])
    make_data_file(valid_samples, valid_labels, "data/valid.jsonl")
    test_samples = []
    test_labels = []
    for idx in indices[int(0.9*length):int(length)]:
        test_samples.append(all_samples[idx])
        test_labels.append(all_labels[idx])
    make_data_file(test_samples, test_labels, "data/test.jsonl")


if __name__ == '__main__':
<<<<<<< HEAD
    main()
=======
    split_dataset("./data")
>>>>>>> 07a9fca9cd4d44048f5d7850c8e5d5e16dcbfe3a
