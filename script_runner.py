import subprocess
import torch
from plot import plot_log_histories
import json

lin_k_s = [1]
att_k_s = [1,3]
seed = 42
reuse_schedule_path = "reuse_schedule.json"
model_name = "prajjwal1/bert-tiny" 
# model_name = 'bert-base-uncased'
reuse_schedules = json.load(open(reuse_schedule_path))
att_schedule = reuse_schedules[0]
lin_schedule = reuse_schedules[1]
file_name = "resprop_warmup_both.png"
batch_size = 128
num_epochs = 15
train_samples = 32000
test_samples = 1000
output_dir = "outputs"

def get_schedule_string(schedule):
    return ','.join(f'(rp: {x}, start: {y})' for x, y in schedule)

log_histories = {}

world_size = torch.cuda.device_count()

print("Number of GPUs:", torch.cuda.device_count())
for idx in range(len(att_schedule)):
    att_str = get_schedule_string(att_schedule[idx])
    lin_str = get_schedule_string(lin_schedule[idx])

    for att_k in att_k_s:
        for lin_k in lin_k_s:
            print(f"\n==== Running att schedule {att_str} lin schedule {lin_str} | att_k={att_k}, lin_k={lin_k} ====\n")

            subprocess.run([
                "torchrun",
                f"--nproc-per-node={world_size}",
                "training_script.py",
                f"--reuse_schedule_idx={idx}",
                f"--att_k={att_k}",
                f"--lin_k={lin_k}",
                f"--seed={seed}",
                f"--model_name={model_name}",
                f"--num_train={train_samples}",
                f"--num_test={test_samples}",
                f"--batch_size={batch_size}",
                f"--num_epochs={num_epochs}"


            ], check=True)

            log_name = f"{output_dir}/log_history_att_{att_str}_lin_{lin_str}_seed_{seed}_att_k_{att_k}_lin_k_{lin_k}.pt"
            key = f"lin_{lin_str}_att_{att_str}_seed_{seed}_att_k_{att_k}_lin_k_{lin_k}"
            log_histories[key] = torch.load(log_name)

# 📈 Plot all histories
plot_log_histories(log_histories, file_name="resprop_warmup_both.png")
print("✅ Plot saved as resprop_warmup_both.png")


