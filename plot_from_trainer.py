#!/usr/bin/env python3
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

def load_log_history(trainer_state_path):
    """
    Load HuggingFace TrainerState JSON and return its log_history list.
    """
    with open(trainer_state_path, 'r') as f:
        state = json.load(f)
    history = state.get("log_history", [])
    if not history:
        raise ValueError(f"No log_history found in {trainer_state_path}")
    return history

def plot_loss(history, output_path=None, label='Baseline'):
    # extract train-loss entries
    train = [(e["step"], e["loss"]) for e in history if "loss" in e]
    steps_tr, loss_tr = zip(*train) if train else ([], [])

    # extract eval-loss entries
    evals = [(e["step"], e["eval_loss"]) for e in history if "eval_loss" in e]
    steps_ev, loss_ev = zip(*evals) if evals else ([], [])

    plt.figure(figsize=(8,5))
    if steps_tr:
        plt.plot(steps_tr, loss_tr, label="train loss")
    if steps_ev:
        plt.plot(steps_ev, loss_ev, label="eval loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training & Evaluation Loss for " + label)
    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

def plot(input_path, output_path):
    history = load_log_history(input_path)
    plot_loss(history, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Plot train/eval loss from a HuggingFace trainer_state.json"
    )
    parser.add_argument(
        "--trainer_state",
        type=Path,
        help="Path to trainer_state.json (in your output_dir)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="If given, save the figure to this file (e.g. loss.png)"
    )
    args = parser.parse_args()

    history = load_log_history(args.trainer_state)
    plot_loss(history, args.output)

if __name__ == "__main__":
    main()