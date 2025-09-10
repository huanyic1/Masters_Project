import json

import matplotlib.pyplot as plt
import numpy as np
import csv


def plot_log_histories(log_histories, file_name=None):
    fig, ax = plt.subplots(2, 1, figsize=(10, 12), dpi=120)

    for reuse_percentage, log_history in log_histories.items():
        log_history = [x for x in log_history]
        # label = "Baseline" if reuse_percentage == 0 else f"RP {int(reuse_percentage * 100)}% K={k}"
        label = reuse_percentage

        loss_x = [x["epoch"] for x in log_history if "loss" in x]
        loss_y = [x["loss"] for x in log_history if "loss" in x]

        acc_x = [x["epoch"] for x in log_history if "eval_accuracy" in x]
        acc_y = [x["eval_accuracy"] for x in log_history if "eval_accuracy" in x]

        ax[0].plot(loss_x, loss_y, label=label, alpha=0.7)
        ax[1].plot(acc_x, acc_y, label=label, alpha=0.7)

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



def export_stats(log_histories, csv_file):
    """
    Exports the final training loss and validation accuracy for each run
    in log_histories to a CSV file.
    
    Parameters:
        log_histories (dict): mapping from reuse_percentage (or label) to list of log entries
        csv_file (str): path to output CSV file
    """
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "Final Loss", "Final Eval Accuracy"])

        for label, log_history in log_histories.items():
            # Filter entries
            loss_entries = [x["loss"] for x in log_history if "loss" in x]
            acc_entries = [x["eval_accuracy"] for x in log_history if "eval_accuracy" in x]

            final_loss = loss_entries[-1] if loss_entries else None
            final_acc = acc_entries[-1] if acc_entries else None

            writer.writerow([label, final_loss, final_acc])

    print(f"Final metrics written to {csv_file}")


if __name__ == "__main__":
    log_histories = {}

    for rp in [0, 50, 70, 90, 95, 99]:
        with open(f"trainer_out/prajjwal1-bert-tiny/rp_{rp}/checkpoint-32000/trainer_state.json", "r") as f:
            state = json.load(f)
            log_histories[rp / 100] = state["log_history"]

    plot_log_histories(log_histories)
