# AUS HPC with VS Code, Remote SSH, Slurm, and custom Python environment

## Purpose

This note records the exact setup path that worked for my AUS HPC account using:

- local laptop
- AUS VPN
- VS Code Remote SSH
- `hpc-gpu` host
- shared storage under `/shared/ihafez`
- a custom Conda environment at `/shared/ihafez/.conda/nn311`
- direct `srun` commands instead of the missing helper script

It also records the important failures and how they were resolved, so the setup can be repeated on another computer later.

---

## What worked in my case

### Important account specific notes

1. I could connect to `hpc-gpu`.
2. I did **not** have access to `hpc-gen`.
3. The guide mentions helper scripts such as `/usr/local/bin/hpc-gpu`, but on my account and host that file was not available.
4. Because of that, the final working workflow used raw `srun` commands directly.
5. The project had to be run from `/shared/...`, not from `/home/...`, so compute nodes could access the files.

---

## Folder layout that worked

### Shared project folder

```bash
/shared/ihafez/nn_training
```

### Shared test folder

```bash
/shared/ihafez/vscode-test
```

### Custom Conda environment

```bash
/shared/ihafez/.conda/nn311
```

---

## Part 1. One time laptop setup in VS Code

### Step 1. Connect to AUS VPN

Connect the AUS VPN first.

Expected result:

- VPN shows connected

### Step 2. Install the VS Code extension

Install:

- `Remote - SSH`

### Step 3. Enable PTY allocation

Open the VS Code user settings JSON and add:

```json
{
    "remote.SSH.permitPtyAllocation": true
}
```

If the file already contains other settings, add only this line inside the same JSON object.

Example final settings JSON:

```json
{
    "editor.fontWeight": "normal",
    "scm.inputFontSize": 14,
    "editor.scrollbar.horizontalScrollbarSize": 13,
    "editor.scrollbar.verticalScrollbarSize": 15,
    "remote.SSH.remotePlatform": {
        "192.168.0.70": "linux",
        "192.168.0.35": "linux"
    },
    "security.workspace.trust.untrustedFiles": "open",
    "editor.minimap.sectionHeaderFontSize": 10,
    "editor.fontSize": 15.25,
    "debug.onTaskErrors": "abort",
    "terminal.integrated.commandsToSkipShell": [
        "matlab.interrupt"
    ],
    "workbench.colorTheme": "Visual Studio Light",
    "json.schemas": [],
    "remote.SSH.permitPtyAllocation": true
}
```

### Step 4. Add the SSH host config

Open the SSH config file on the laptop, usually:

```text
C:\Users\<your-windows-username>\.ssh\config
```

Add this:

```ssh
Host 192.168.0.35
  HostName 192.168.0.35
  User pi

Host hpc-gen
  HostName hpc-gen.aus.edu
  User ihafez
  ServerAliveInterval 60

Host hpc-gpu
  HostName hpc-gpu.aus.edu
  User ihafez
  ServerAliveInterval 60
```

Notes:

- The old Raspberry Pi host can remain.
- The important host for my account was `hpc-gpu`.
- `hpc-gen` was listed in config, but I did not have access to it.

### Step 5. Connect from VS Code

Use:

```text
Remote-SSH: Connect to Host
```

Select:

```text
hpc-gpu
```

Enter the AUS HPC password when prompted.

Expected result:

- the bottom left of VS Code shows `SSH: hpc-gpu`

### Step 6. Verify remote connection in the terminal

Run:

```bash
pwd
whoami
hostname
```

Expected result:

- `whoami` should be `ihafez`
- `hostname` should be an HPC machine name such as `ip-10-240-16-218`

---

## Part 2. Move the project to shared storage

### Why this was needed

The code was originally in:

```bash
/home/ihafez/Documents/nn_training
```

The final working workflow used:

```bash
/shared/ihafez/nn_training
```

### Copy the project once

Run:

```bash
mkdir -p /shared/ihafez/nn_training
cp -ru /home/ihafez/Documents/nn_training/* /shared/ihafez/nn_training/
cd /shared/ihafez/nn_training
pwd
```

Expected result:

```bash
/shared/ihafez/nn_training
```

---

## Part 3. Confirm Slurm and partitions

### Step 1. Check Slurm commands

Run:

```bash
which srun
which sbatch
which squeue
which sinfo
```

Expected result in my case:

```bash
/opt/slurm/bin/srun
/opt/slurm/bin/sbatch
/opt/slurm/bin/squeue
/opt/slurm/bin/sinfo
```

### Step 2. Check available partitions

Run:

```bash
sinfo
```

Expected important result in my case:

- `gpu` partition existed
- default partition was `cpu*`

This mattered because if `-p gpu` is omitted, Slurm may try the CPU partition and reject a GPU request.

---

## Part 4. First GPU test and what failed

### Failure 1. Missing helper script

This command failed:

```bash
cp /usr/local/bin/hpc-gpu .
```

Observed error:

```bash
cp: cannot stat '/usr/local/bin/hpc-gpu': No such file or directory
```

Interpretation:

- the helper script path from the guide was not usable on my accessible host
- the final workflow therefore used raw `srun` directly

### Failure 2. Using `srun` without the GPU partition

This command failed:

```bash
srun --gres=gpu:1 --time=00:05:00 --cpus-per-task=2 --mem=8G --pty bash -l
```

Observed error:

```bash
srun: error: Unable to allocate resources: Requested node configuration is not available
```

Reason:

- the command did not specify `-p gpu`

### First working interactive GPU allocation

This command worked:

```bash
srun -p gpu --gres=gpu:1 --time=00:05:00 --cpus-per-task=2 --mem=8G --pty bash -l
```

What happened:

- the job first went to `CF` state
- then to `R`
- the shell landed on a compute node such as:

```bash
gpu-dy-g5-0-30
```

How to monitor:

```bash
squeue -u ihafez
```

Example observed states:

```text
JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)
4008  gpu       bash ihafez CF 3:01 1 gpu-dy-g5-0-30
```

then later:

```text
JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)
4008  gpu       bash ihafez R 4:43 1 gpu-dy-g5-0-30
```

### Failure 3. Interactive session died due to time limit

Observed message:

```text
slurmstepd: error: *** STEP 4008.0 ON gpu-dy-g5-0-30 CANCELLED ... DUE TO TIME LIMIT ***
srun: error: gpu-dy-g5-0-30: task 0: Killed
```

Interpretation:

- the allocation worked
- the time limit expired before I used the shell enough

### Better approach chosen after that

Use direct non interactive `srun` for Python scripts instead of opening an interactive shell first.

---

## Part 5. Python environments that failed

### Existing environment 1

Test:

```bash
/opt/miniconda/envs/ase-py311/bin/python -c "import torch, numpy, sklearn, matplotlib; print('basic imports ok')"
```

Observed error:

```text
ModuleNotFoundError: No module named 'torch'
```

### Existing environment 2

Test:

```bash
/opt/miniconda/envs/gurobi-gpu-py311/bin/python -c "import os; import numpy as np; import matplotlib.pyplot as plt; import torch; import torch.nn as nn; import torch.optim as optim; from torch.utils.data import DataLoader, TensorDataset; from sklearn.preprocessing import StandardScaler; print('all imports ok')"
```

Observed error:

```text
ModuleNotFoundError: No module named 'numpy'
```

### Conclusion

Neither existing environment was correct for the project.

---

## Part 6. `venv` attempt that failed

### Failed attempt

```bash
python3 -m venv /shared/ihafez/.venvs/nn311
```

Observed error:

```text
Error: Command '['/shared/ihafez/.venvs/nn311/bin/python3', '-m', 'ensurepip', '--upgrade', '--default-pip']' returned non-zero exit status 1.
```

### Conclusion

Do not use `venv` here.
Use Conda instead.

---

## Part 7. Final working Python environment

### Step 1. Initialize Conda in the shell

Run:

```bash
source /opt/miniconda/etc/profile.d/conda.sh
conda --version
```

Expected result:

- a Conda version is printed

### Step 2. Create the custom environment in shared storage

Run:

```bash
conda create -y -p /shared/ihafez/.conda/nn311 python=3.11 numpy matplotlib scikit-learn pip
```

Expected result:

- environment created successfully at `/shared/ihafez/.conda/nn311`

### Step 3. Activate it

Run:

```bash
conda activate /shared/ihafez/.conda/nn311
```

Expected result:

- shell prompt begins with:

```text
(/shared/ihafez/.conda/nn311)
```

### Step 4. First PyTorch install that partially worked but failed on GPU

Initial command used:

```bash
python -m pip install torch torchvision torchaudio
```

This installed a CUDA 13.0 build.

Import test passed, but GPU test failed.

### Import test that passed

```bash
python -c "import os; import numpy as np; import matplotlib.pyplot as plt; import torch; import torch.nn as nn; import torch.optim as optim; from torch.utils.data import DataLoader, TensorDataset; from sklearn.preprocessing import StandardScaler; print('ALL_IMPORTS_OK'); print('torch', torch.__version__)"
```

Observed result:

```text
ALL_IMPORTS_OK
torch 2.11.0+cu130
```

### GPU test that failed because of the driver mismatch

```bash
srun -p gpu --gres=gpu:1 --time=00:15:00 --cpus-per-task=2 --mem=8G /shared/ihafez/.conda/nn311/bin/python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpus', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
```

Observed result:

```text
torch 2.11.0+cu130
cuda False
gpus 1
NO_GPU
```

There was also a warning about the NVIDIA driver being too old for that build.

### Step 5. Replace PyTorch with CUDA 12.8 build

Run:

```bash
conda activate /shared/ihafez/.conda/nn311
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

### Step 6. Final GPU verification that worked

Run:

```bash
srun -p gpu --gres=gpu:1 --time=00:15:00 --cpus-per-task=2 --mem=8G /shared/ihafez/.conda/nn311/bin/python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpus', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
```

Observed successful result:

```text
torch 2.10.0+cu128
cuda True
gpus 1
NVIDIA A10G
```

This was the final confirmation that the environment and GPU path worked.

---

## Part 8. First project script tested

### Script file

```bash
/shared/ihafez/nn_training/C54_nonlinear_z_pid_WFdBk.py
```

### Important script behavior

This script:

1. performs nonlinear PID simulation
2. trains a GRU
3. compares the GRU controller against PID
4. saves figures beside the script
5. saves a trained model into a `models` folder beside the script
6. contains multiple `plt.show()` calls

### Why `MPLBACKEND=Agg` was needed

Because the script calls `plt.show()`, the noninteractive backend was forced during compute node execution.

### Resource request that failed

This command failed:

```bash
srun -p gpu --gres=gpu:1 --time=04:00:00 --cpus-per-task=4 --mem=16G /shared/ihafez/.conda/nn311/bin/python C54_nonlinear_z_pid_WFdBk.py > C54_run.log 2>&1
```

Observed error in `C54_run.log`:

```text
srun: error: Memory specification can not be satisfied
srun: error: Unable to allocate resources: Requested node configuration is not available
```

### Final working run template for this project

Use this smaller request first:

```bash
cd /shared/ihafez/nn_training
export MPLBACKEND=Agg
srun -p gpu --gres=gpu:1 --time=04:00:00 --cpus-per-task=2 --mem=8G /shared/ihafez/.conda/nn311/bin/python C54_nonlinear_z_pid_WFdBk.py > C54_run.log 2>&1
```

### Monitor the run

Open another terminal and run:

```bash
squeue -u ihafez
```

and:

```bash
tail -f /shared/ihafez/nn_training/C54_run.log
```

### If the command returns immediately

That is not necessarily a failure.
Because output is redirected to the log file, check:

```bash
ls -lh /shared/ihafez/nn_training/C54_run.log
tail -n 50 /shared/ihafez/nn_training/C54_run.log
squeue -u ihafez
```

### Expected output locations for this script

Because the script defines:

```python
FIG_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(FIG_DIR, "models")
```

expected outputs are saved in:

```bash
/shared/ihafez/nn_training
```

and model files in:

```bash
/shared/ihafez/nn_training/models
```

### Common files to check later

```bash
ls -lh /shared/ihafez/nn_training
ls -lh /shared/ihafez/nn_training/models
```

---

## Part 9. General commands that were useful

### Check the queue

```bash
squeue -u ihafez
```

### Check partitions

```bash
sinfo
```

### Check log output

```bash
tail -n 50 /shared/ihafez/nn_training/C54_run.log
```

### Watch log output live

```bash
tail -f /shared/ihafez/nn_training/C54_run.log
```

### Check current folder

```bash
pwd
```

### Check generated files

```bash
ls -lh /shared/ihafez/nn_training
ls -lh /shared/ihafez/nn_training/models
```

---

## Part 10. Clean restart routine after laptop restart

This section is the short procedure to continue later after restarting the laptop.

### What does not need to be repeated

You do **not** need to repeat:

1. creation of the Conda environment
2. package installation
3. VS Code PTY JSON setting on the same laptop
4. SSH host config on the same laptop

### Restart procedure

#### Step 1. Connect AUS VPN

Connect the VPN first.

#### Step 2. Open VS Code

Open VS Code on the laptop.

#### Step 3. Connect with Remote SSH

Use:

```text
Remote-SSH: Connect to Host
```

Select:

```text
hpc-gpu
```

Enter the password.

Expected result:

- bottom left shows `SSH: hpc-gpu`

#### Step 4. Open the shared project folder

Open:

```bash
/shared/ihafez/nn_training
```

#### Step 5. Open a terminal and restore the shell environment

Run:

```bash
cd /shared/ihafez/nn_training
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/ihafez/.conda/nn311
export MPLBACKEND=Agg
```

Expected result:

- the prompt begins with:

```text
(/shared/ihafez/.conda/nn311)
```

#### Step 6. If you only want to verify the environment again

Run:

```bash
/shared/ihafez/.conda/nn311/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

This checks the Python environment, but note that GPU availability is only meaningful on an allocated GPU compute node.

#### Step 7. If you want to retest GPU quickly

Run:

```bash
srun -p gpu --gres=gpu:1 --time=00:15:00 --cpus-per-task=2 --mem=8G /shared/ihafez/.conda/nn311/bin/python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpus', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
```

Expected successful result:

```text
torch 2.10.0+cu128
cuda True
gpus 1
NVIDIA A10G
```

#### Step 8. Run the project again

```bash
cd /shared/ihafez/nn_training
export MPLBACKEND=Agg
srun -p gpu --gres=gpu:1 --time=04:00:00 --cpus-per-task=2 --mem=8G /shared/ihafez/.conda/nn311/bin/python C54_nonlinear_z_pid_WFdBk.py > C54_run.log 2>&1
```

#### Step 9. Monitor status and logs

```bash
squeue -u ihafez
tail -f /shared/ihafez/nn_training/C54_run.log
```

#### Step 10. Check whether a job survived or ended after disconnect or restart

After reconnecting, always check:

```bash
squeue -u ihafez
tail -n 50 /shared/ihafez/nn_training/C54_run.log
```

This tells whether the job is still in the queue and what the latest log output says.

---

## Part 11. Minimal copy paste blocks

### Minimal reconnect and run block

```bash
cd /shared/ihafez/nn_training
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/ihafez/.conda/nn311
export MPLBACKEND=Agg
srun -p gpu --gres=gpu:1 --time=04:00:00 --cpus-per-task=2 --mem=8G /shared/ihafez/.conda/nn311/bin/python C54_nonlinear_z_pid_WFdBk.py > C54_run.log 2>&1
```

### Minimal reconnect and monitor block

```bash
cd /shared/ihafez/nn_training
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/ihafez/.conda/nn311
squeue -u ihafez
tail -f /shared/ihafez/nn_training/C54_run.log
```

### Minimal GPU check block

```bash
cd /shared/ihafez/nn_training
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/ihafez/.conda/nn311
srun -p gpu --gres=gpu:1 --time=00:15:00 --cpus-per-task=2 --mem=8G /shared/ihafez/.conda/nn311/bin/python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpus', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
```

---

## Part 12. Most important lessons from this setup

1. Use `/shared/ihafez/...` for the project and outputs.
2. On my account, connect to `hpc-gpu`.
3. Do not rely on `/usr/local/bin/hpc-gpu` helper script unless it is actually present.
4. Use `srun -p gpu ...` directly.
5. The default partition is not the GPU partition, so always specify `-p gpu`.
6. The final working Python environment is:

```bash
/shared/ihafez/.conda/nn311
```

7. The final working PyTorch build was:

```text
torch 2.10.0+cu128
```

8. The final GPU test result was:

```text
cuda True
NVIDIA A10G
```

9. If a script contains `plt.show()`, run with:

```bash
export MPLBACKEND=Agg
```

10. If Slurm rejects memory or CPU requests, reduce them and try again.

