import os
# import tensorflow as tf
import pickle
import gzip
from collections import namedtuple
# import GPUtil
import yaml


def pickle_object(filepath, obj):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickled_object(filepath, gzipped=False):
    if gzipped:
        with gzip.open(filepath, "rb") as f:
            obj = pickle.load(f)
    else:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)

    return obj


# def flatten(x):
#     N = x.shape[0]
#     return tf.reshape(x, (N, -1))

def get_gpu_utilization_info():
    gpu_strings = GPUtil.showUtilization()
    info = {}
    for gpu_string in gpu_strings:
        gpu_info = gpu_string.replace(" ", "").replace("%", "").split("|")[1:-1]
        gpu_id = int(gpu_info[0])
        gpu_utilization = float(gpu_info[1]) / 100
        gpu_memory_usage = float(gpu_info[2]) / 100

        info[gpu_id] = {"utilization":gpu_utilization, "mem_usage":gpu_memory_usage}
    return info


def save_args(args, trial_dir):
    pickle_object(os.path.join(trial_dir, "args.pkl"), args)

    with open(os.path.join(trial_dir, "args.yml"), "w") as f:
        yaml.dump(args.__dict__, f)
