import json

import matplotlib.pyplot as plt
import numpy as np


def plot_log_histories(log_histories, file_name=None):
    fig, ax = plt.subplots(2, 1, figsize=(10, 12), dpi=120)

    for reuse_percentage, log_history in log_histories.items():
        log_history = [x for x in log_history]
        label = "Baseline" if reuse_percentage == 0 else f"RP {int(reuse_percentage * 100)}%"

        loss_x = [x["epoch"] for x in log_history if "loss" in x]
        loss_y = [x["loss"] for x in log_history if "loss" in x]

        acc_x = [x["epoch"] for x in log_history if "eval_accuracy" in x]
        acc_y = [x["eval_accuracy"] for x in log_history if "eval_accuracy" in x]

        ax[0].plot(loss_x, loss_y, label=label)
        ax[1].plot(acc_x, acc_y, label=label)

    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Training Loss")

    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Validation Accuracy")

    ax[0].grid(alpha=0.4)
    ax[1].grid(alpha=0.4)

    ax[1].set_yscale('function', functions=(
        lambda a: np.exp(a * 5),
        lambda a: np.log(a) / 5
    ))

    ax[0].legend()
    ax[1].legend()

    if file_name:
        plt.savefig(file_name)
    plt.show()


if __name__ == "__main__":
    log_histories = {}

    for rp in [0, 50, 70, 90, 95, 99]:
        with open(f"trainer_out/prajjwal1-bert-tiny/rp_{rp}/checkpoint-32000/trainer_state.json", "r") as f:
            state = json.load(f)
            log_histories[rp / 100] = state["log_history"]

    plot_log_histories(log_histories)
