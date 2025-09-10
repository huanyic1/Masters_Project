import subprocess
import torch
from plot import plot_log_histories, export_stats
import json
import random


lin_k_s = [1]
att_k_s = [1]
seed = 42
reuse_schedule_path = "reuse_schedule_finetune.json"
# model_name = "prajjwal1/bert-tiny" 
model_name = 'bert-base-uncased'
reuse_schedules = json.load(open(reuse_schedule_path))
att_schedule = reuse_schedules[0]
lin_schedule = reuse_schedules[1]
file_name = "resprop_warmup_both.png"
batch_size = 128
num_epochs = 1
train_samples = 32000
test_samples = 1000
output_dir = "outputs_finetune"
# output_dir = "junk_outputs"

baseline = False
att = True

plot_name = "resprop_warmup_both.png"
def get_schedule_string(schedule):
    return ','.join(f'(rp: {x}, start: {y})' for x, y in schedule)

port = random.randint(10000, 60000)

log_histories = {}

world_size = torch.cuda.device_count()

print("Number of GPUs:", torch.cuda.device_count())

if baseline: 
    log_name = f"{output_dir}/log_history_baseline_seed_{seed}.pt"
    print(f"\n==== Running Baseline====\n")
    subprocess.run([
        "torchrun",
        f"--master-port={port}",
        f"--nproc-per-node={world_size}",
        "finetune_script.py",
        f"--reuse_schedule_idx={0}",
        f"--seed={seed}",
        f"--model_name={model_name}",
        f"--num_train={train_samples}",
        f"--num_test={test_samples}",
        f"--batch_size={batch_size}",
        f"--num_epochs={num_epochs}", 
        "--baseline",
        f"--log_name={log_name}",
    ], check=True)
    key = f"baseline"
    log_histories[key] = torch.load(log_name)

# keys = ["rp: 0", "rp: 50%, 25% Warmup", "rp: 75%, 25% Warmup ", "rp: 90%, 25% Warmup", "rp: 99%, 25% Warmup"]
keys = ["rp: 0", "rp: 2:4, no Warmup", "rp: 2:4, 25% Warmup ", "rp: 1:4, no Warmup", "rp: 1:4, 25% Warmup"]
for idx in range(len(att_schedule)):
    att_str = get_schedule_string(att_schedule[idx])
    lin_str = get_schedule_string(lin_schedule[idx])

    if att:
        log_name = f"{output_dir}/log_history_att_{att_str}_lin_{lin_str}_seed_{seed}_new.pt"
        key = f"lin_{lin_str}_att_{att_str}"
    else: 
        log_name = f"{output_dir}/log_history_lin_{lin_str}_seed_{seed}_new.pt"
        key = f"lin_{lin_str}" 
    key = keys[idx]
    print(f"\n==== Running att schedule {att_str} lin schedule {lin_str}====\n")
    # args = [
    #         "torchrun",
    #         f"--master-port={port}",
    #         f"--nproc-per-node={world_size}",
    #         "finetune_script.py",
    #         f"--reuse_schedule_idx={idx}",
    #         f"--seed={seed}",
    #         f"--model_name={model_name}",
    #         f"--num_train={train_samples}",
    #         f"--num_test={test_samples}",
    #         f"--batch_size={batch_size}",
    #         f"--num_epochs={num_epochs}", 
    #         f"--log_name={log_name}",
    #         f"--output_dir={output_dir}"
    #     ]
    # if att:
    #     args.append("--att")

    # subprocess.run(args, check=True)
    log_histories[key] = torch.load(log_name)
# 📈 Plot all histories
plot_log_histories(log_histories, file_name=plot_name)
export_stats(log_histories, "finetune_stats.csv")
print(f"Plot saved as {plot_name}")


