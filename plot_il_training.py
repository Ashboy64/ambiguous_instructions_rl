import numpy as np
import pandas
import os
from matplotlib import pyplot as plt
from glob import glob

LOGDIR = "/iris/u/khatch/CS224N/ambiguous_instructions_rl/logs"
# ENV_NAME = "GoToLocal"
# ENV_NAME = "OpenDoorLoc"
# ENV_NAME = "PickupLoc"
ENV_NAMES = ["GoToLocal", "OpenDoorLoc", "PickupLoc", "PutNextLocalS6N4"]

def make_plots(logdir, env_names):
    for env_name in env_names:
        print(f"{env_name}")
        make_plot(logdir, env_name)


def make_plot(logdir, env_name):
    for csv_file in glob(os.path.join(logdir, f"BabyAI-{env_name}-v0_*", "log.csv")):
        # print(f"Plotting data from \"{csv_file}\"...")
        header = get_header(csv_file)
        frames = np.loadtxt(csv_file, delimiter=",", usecols=header.index("frames"), skiprows=1)
        validation_success_rate = np.loadtxt(csv_file, delimiter=",", usecols=header.index("validation_success_rate"), skiprows=1)

        if "plain" in csv_file:
            name = "Unambiguous"
            color = "tab:red"
        elif "0.5" in csv_file:
            name = "50% Ambiguity"
            color = "tab:orange"
        elif "1.0" in csv_file:
            name = "100% Ambiguity"
            color = "tab:blue"
        elif "nonsense" in csv_file:
            name = "Nonsense Instructions"
            color = "tab:green"
        else:
            assert False

        plt.plot(frames, validation_success_rate, label=name, color=color)

    plt.title(env_name)
    plt.ylabel("Success Rate")
    plt.xlabel("Training Step")
    plt.legend(loc="lower right")
    plt.ylim([-0.1, 1.1])
    plot_path = os.path.join("plots", f"{env_name}_IL_training.png")

    if not os.path.isdir(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))

    plt.savefig(plot_path)#, dpi=100)
    plt.clf()


def get_header(filename):
    with open(filename, "r") as f:
        header_line = f.readline().strip("\n")
        header = header_line.split(",")
        return header


if __name__ == "__main__":
    make_plots(LOGDIR, ENV_NAMES)
