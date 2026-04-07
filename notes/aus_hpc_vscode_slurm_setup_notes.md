# AUS HPC with VS Code and Slurm from Laptop

## Scope
This note records the exact workflow that worked in our session up to the current stage.

It is written for later reuse on another computer.

It is based on two things:

1. The AUS IT guide for using VS Code with AUS HPC through Remote SSH and Slurm.
2. The account specific behavior observed in this session.

## What this workflow is for
Use VS Code on your own laptop to:

1. connect to AUS HPC by Remote SSH,
2. edit files on the remote system,
3. keep project files under `/shared/...`,
4. submit GPU work through Slurm,
5. avoid running training directly on the login session.

## Important account specific note
The official guide shows a workflow that starts from `hpc-gen` and uses helper scripts such as `/usr/local/bin/hpc-gpu`.

In this session, the account had access to `hpc-gpu` only, not `hpc-gen`.

Also, on the connected host, `/usr/local/bin/hpc-gpu` was not present.

Because of that, the working method in this session was:

1. connect by Remote SSH to `hpc-gpu`,
2. work under `/shared/...`,
3. use raw Slurm commands such as `srun` instead of the missing helper script.

## Step 1. Connect to AUS VPN
Before opening the remote HPC host, connect to AUS VPN with AWS Client VPN.

Checklist:

1. Open AWS Client VPN on the laptop.
2. Use the AUS profile.
3. Complete AUS SSO if prompted.
4. Confirm the VPN status shows `Connected`.

## Step 2. Install the needed VS Code extension
Install the VS Code extension:

```text
Remote - SSH
```

## Step 3. Enable Permit PTY Allocation
The UI search did not show the setting properly, so the working method was through `settings.json`.

Open User Settings JSON and make sure the following line exists:

```json
"remote.SSH.permitPtyAllocation": true
```

Example:

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

## Step 4. Add AUS HPC hosts to SSH config
Open the local SSH config file on Windows:

```text
C:\Users\<your-windows-username>\.ssh\config
```

The working content in this session was:

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

For another account or another user, replace `ihafez` with the actual AUS HPC username.

General template:

```ssh
Host hpc-gen
  HostName hpc-gen.aus.edu
  User YOUR_AUS_HPC_USERNAME
  ServerAliveInterval 60

Host hpc-gpu
  HostName hpc-gpu.aus.edu
  User YOUR_AUS_HPC_USERNAME
  ServerAliveInterval 60
```

## Step 5. Connect by Remote SSH
In VS Code:

1. `Ctrl + Shift + P`
2. run `Remote-SSH: Connect to Host`
3. choose `hpc-gpu`
4. enter the AUS HPC password

In this session, connection to `hpc-gpu` succeeded.

Bottom left status showed:

```text
SSH: hpc-gpu
```

## Step 6. Confirm the remote session
Run:

```bash
pwd
whoami
hostname
```

Observed output in this session:

```text
/home/ihafez/Documents/nn_training
ihafez
ip-10-240-16-218
```

This confirmed that VS Code was connected to the remote HPC environment.

## Step 7. Use shared storage for the project
Even if the original project is under `/home/...`, the working directory for Slurm jobs should be under `/shared/...`.

Create the shared project folder:

```bash
mkdir -p /shared/ihafez/nn_training
```

Copy the project into shared storage:

```bash
cp -r /home/ihafez/Documents/nn_training/* /shared/ihafez/nn_training/
```

Enter the shared project folder:

```bash
cd /shared/ihafez/nn_training
pwd
```

Observed output:

```text
/shared/ihafez/nn_training
```

For a small isolated test folder, this also worked:

```bash
mkdir -p /shared/ihafez/vscode-test
cd /shared/ihafez/vscode-test
pwd
```

Observed output:

```text
/shared/ihafez/vscode-test
```

## Step 8. Select the remote Python interpreter in VS Code
VS Code first tried to use a local Windows interpreter, which was invalid for the remote session.

Do not use a local Windows path such as:

```text
C:\Users\ishaq\AppData\Local\Programs\Python\Python313\python.exe
```

In this session, the selected remote interpreter was:

```text
Python 3.11.14 ('ase-py311')  /opt/miniconda/envs/ase-py311/bin/python
```

This fixed the remote interpreter issue inside VS Code.

## Step 9. Create a GPU test script
Inside `/shared/ihafez/vscode-test`, create `gpu_test.py` with the following content:

```python
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    x = torch.randn(2000, 2000, device="cuda")
    y = torch.randn(2000, 2000, device="cuda")
    z = x @ y
    print("GPU matmul done. Shape:", z.shape)
else:
    print("CUDA is not available")
```

## Step 10. Check Slurm commands and available partitions
The helper script route could not be used because `/usr/local/bin/hpc-gpu` was not present on the accessible host.

Check Slurm tools:

```bash
which srun
which sbatch
which squeue
which sinfo
```

Observed paths:

```text
/opt/slurm/bin/srun
/opt/slurm/bin/sbatch
/opt/slurm/bin/squeue
/opt/slurm/bin/sinfo
```

Check cluster partitions:

```bash
sinfo
```

Observed relevant output:

```text
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
cpu*         up   infinite     54  idle~ cpu-dy-c5-0-[7-60]
gpu          up   infinite     53  idle~ gpu-dy-g5-0-[30,34,40-90]
gpu          up   infinite     31    mix gpu-dy-g5-0-[1-16,18-29,31-33]
gpu          up   infinite      6  alloc gpu-dy-g5-0-[17,35-39]
```

Important conclusion:

The default partition was `cpu`, so GPU jobs must explicitly request:

```text
-p gpu
```

## Step 11. First failed GPU allocation attempt
This command failed because it did not specify the `gpu` partition:

```bash
srun --gres=gpu:1 --time=00:05:00 --cpus-per-task=2 --mem=8G --pty bash -l
```

Observed error:

```text
srun: error: Unable to allocate resources: Requested node configuration is not available
```

## Step 12. Correct interactive Slurm allocation
This command was accepted:

```bash
cd /shared/ihafez/vscode-test
srun -p gpu --gres=gpu:1 --time=00:05:00 --cpus-per-task=2 --mem=8G --pty bash -l
```

What happened:

1. the terminal initially looked idle,
2. the job appeared in `squeue`,
3. the state first became `CF`,
4. later it became `R`.

Queue check command:

```bash
squeue -u ihafez
```

Observed state while configuring:

```text
JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)
4008 gpu bash ihafez CF 3:01 1 gpu-dy-g5-0-30
```

Observed state while running:

```text
JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)
4008 gpu bash ihafez R 4:43 1 gpu-dy-g5-0-30
```

The compute node shell prompt was:

```text
ihafez@gpu-dy-g5-0-30:/shared/ihafez/vscode-test$
```

This confirmed that Slurm had opened a shell on a compute node.

## Step 13. Why the interactive job ended
The interactive job was killed because the time limit was only 5 minutes.

Observed message:

```text
srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
slurmstepd: error: *** STEP 4008.0 ON gpu-dy-g5-0-30 CANCELLED AT 2026-04-06T14:04:38 DUE TO TIME LIMIT ***
srun: error: gpu-dy-g5-0-30: task 0: Killed
```

So the interactive allocation itself worked.

The failure was not a connection problem.

The job simply reached its time limit before the test script was run to completion.

## Step 14. Recommended direct test command
For a simple verification, use a direct non interactive `srun` command instead of opening an interactive shell.

Recommended command:

```bash
cd /shared/ihafez/vscode-test
srun -p gpu --gres=gpu:1 --time=00:15:00 --cpus-per-task=2 --mem=8G /opt/miniconda/envs/ase-py311/bin/python gpu_test.py
```

This is the next command to run.

It should execute the test directly on a GPU compute node and print the result back to the terminal.

## Step 15. General pattern for the real training script
After the GPU test works, use the same pattern for the actual training code.

Example:

```bash
cd /shared/ihafez/nn_training
srun -p gpu --gres=gpu:1 --time=02:00:00 --cpus-per-task=4 --mem=16G /opt/miniconda/envs/ase-py311/bin/python train.py
```

Replace `train.py` with the actual script name if needed.

## Step 16. Commands that should not be used for training
Do not run training directly on the login session with commands such as:

```bash
python train.py
```

or:

```bash
python gpu_test.py
```

outside a Slurm allocation.

## Step 17. Minimal quick start for another computer
If repeating this on another computer, the shortest safe sequence is:

### A. Laptop side
1. connect AWS Client VPN,
2. open VS Code,
3. install `Remote - SSH`,
4. ensure `"remote.SSH.permitPtyAllocation": true` in settings,
5. add `hpc-gpu` to `C:\Users\<user>\.ssh\config`,
6. connect to `hpc-gpu`.

### B. Remote side
Run:

```bash
mkdir -p /shared/YOUR_USERNAME/vscode-test
cd /shared/YOUR_USERNAME/vscode-test
```

Create `gpu_test.py` with the test code.

Then run:

```bash
srun -p gpu --gres=gpu:1 --time=00:15:00 --cpus-per-task=2 --mem=8G /opt/miniconda/envs/ase-py311/bin/python gpu_test.py
```

## Step 18. Current verified status
Verified up to this stage:

1. VPN connection concept established.
2. VS Code Remote SSH connection to `hpc-gpu` worked.
3. `Permit PTY Allocation` was enabled through `settings.json`.
4. Remote Python interpreter was corrected to `ase-py311`.
5. Project files were moved to shared storage.
6. Slurm commands were available.
7. GPU partition name was identified as `gpu`.
8. Interactive Slurm GPU allocation worked.
9. Interactive job timed out only because the time limit was too short.
10. The next correct action is the direct non interactive `srun` test command.

## Step 19. Useful checks
Check current queue:

```bash
squeue -u ihafez
```

Check partitions:

```bash
sinfo
```

Check current machine:

```bash
hostname
```

Check GPU visibility when inside a compute allocation:

```bash
nvidia-smi
```

Check current path:

```bash
pwd
```

## Step 20. One block summary
```bash
# local laptop side
# 1. connect AUS VPN
# 2. connect to hpc-gpu using Remote SSH from VS Code

# remote HPC side
mkdir -p /shared/ihafez/vscode-test
cd /shared/ihafez/vscode-test

# create gpu_test.py in this folder, then run
srun -p gpu --gres=gpu:1 --time=00:15:00 --cpus-per-task=2 --mem=8G /opt/miniconda/envs/ase-py311/bin/python gpu_test.py
```
