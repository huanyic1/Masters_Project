import subprocess
import torch
from plot import plot_log_histories
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
num_epochs = 15
train_samples = 32000
test_samples = 128
output_dir = "outputs_finetune"
baseline = False
plot_name = "resprop_warmup_both.png"
def get_schedule_string(schedule):
    return ','.join(f'(rp: {x}, start: {y})' for x, y in schedule)

port = random.randint(10000, 60000)

log_histories = {}

world_size = torch.cuda.device_count()

print("Number of GPUs:", torch.cuda.device_count())

if baseline: 
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
                    "--baseline=True",
                ], check=True)
    log_name = f"{output_dir}/log_history_baseline_seed_{seed}.pt"
    key = f"baseline"
    log_histories[key] = torch.load(log_name)

for idx in range(len(att_schedule)):
    att_str = get_schedule_string(att_schedule[idx])
    lin_str = get_schedule_string(lin_schedule[idx])

    for att_k in att_k_s:
        for lin_k in lin_k_s:       
            print(f"\n==== Running att schedule {att_str} lin schedule {lin_str}====\n")
            subprocess.run([
                "torchrun",
                f"--master-port={port}",
                f"--nproc-per-node={world_size}",
                "finetune_script.py",
                f"--reuse_schedule_idx={idx}",
                f"--seed={seed}",
                f"--model_name={model_name}",
                f"--num_train={train_samples}",
                f"--num_test={test_samples}",
                f"--batch_size={batch_size}",
                f"--num_epochs={num_epochs}", 
            ], check=True)

            log_name = f"{output_dir}/log_history_att_{att_str}_lin_{lin_str}_seed_{seed}_new.pt"
            key = f"lin_{lin_str}_att_{att_str}_seed_{seed}"
            log_histories[key] = torch.load(log_name)
# 📈 Plot all histories
plot_log_histories(log_histories, file_name=plot_name)
print(f"Plot saved as {plot_name}")


