# Training AMP on the Unitree Go2 with MimicKit

Tutorial for `masters/MimicKit` + `masters/MimicKit_Data`. Everything below is based on the files in your clone
(`args/amp_go2_args.txt`, `data/envs/amp_go2_env.yaml`, `data/agents/amp_go2_agent.yaml`, ...).

---

## 0. Read this first: Isaac Gym vs. your machine

| Your machine | Isaac Gym Preview 4 requirement |
|---|---|
| Windows 11 | Linux only (Ubuntu 18.04/20.04 officially) |
| RTX 5060 Ti (Blackwell, sm_120) | Needs PyTorch built for sm_120 -> torch >= 2.7 / CUDA 12.8 |
| - | Python bindings exist only for Python 3.6 / 3.7 / 3.8 |

PyTorch 2.7+ does not support Python 3.8, and Isaac Gym's bindings do not exist for newer Pythons, so on this GPU you
**cannot realistically get a working Isaac Gym + torch pair**, even inside WSL2. (I have not tried it on your box;
this is from the version constraints. If you want to attempt it anyway, Path A below is the procedure.)

MimicKit is built so the simulator is swappable by one flag (`--engine_config`). The README lists **Newton**
(tested with `v1.0.0`) and **Isaac Lab** as the other backends. **Newton is the practical choice for you** (Warp-based,
runs on Windows and Blackwell with a current PyTorch). Path B covers it. Steps 1, 2, 5, 6 are identical for both paths.

---

## 1. Put the data in the right place

Your downloaded data is at `masters/MimicKit_Data` (assets, logs, models, motions). MimicKit expects it inside
`MimicKit/data/`. Merge it (this does not clobber `agents/`, `envs/`, `engines/`, `datasets/`):

PowerShell:
```powershell
cd C:\Users\hgtll\Documents\Repos\masters
Copy-Item -Recurse -Force .\MimicKit_Data\* .\MimicKit\data\
```

Check the Go2 files are there:
```powershell
dir MimicKit\data\assets\go2      # go2.xml, go2.usd, meshes/
dir MimicKit\data\motions\go2     # go2_pace/run/trot/walk0-3 .pkl
```
There is also a pretrained `deepmimic_go2_pace_model.pt` in `data/models` (DeepMimic, not AMP, but handy for a sim sanity check).

---

## 2. Create the environment

Always use a dedicated conda env per simulator.

### Path A: Isaac Gym (Linux / WSL2 Ubuntu 20.04, older GPUs only)

```bash
conda create -n mimickit-ig python=3.8 -y
conda activate mimickit-ig

# 1. Download IsaacGym_Preview_4_Package.tar.gz from https://developer.nvidia.com/isaac-gym (needs an NVIDIA login)
tar -xzf IsaacGym_Preview_4_Package.tar.gz
pip install -e isaacgym/python

# 2. A torch build matching your GPU (this is the step Blackwell cannot satisfy on py3.8)
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 3. Isaac Gym uses np.float, removed in newer numpy
pip install "numpy==1.23.5"

# 4. MimicKit deps
cd /path/to/masters/MimicKit
pip install -r requirements.txt
```
Verify the simulator alone: `cd isaacgym/python/examples && python joint_monkey.py`.
Engine flag: `--engine_config data/engines/isaac_gym_engine.yaml` (already what the Go2 arg files use).

### Path B: Newton (recommended for your Windows + RTX 5060 Ti)

```powershell
conda create -n mimickit python=3.11 -y
conda activate mimickit

# PyTorch with Blackwell support (any CUDA 12.8+ wheel works; cu132 is the current stable per pytorch.org,
# your driver reports CUDA 13.4). torchvision is not needed.
pip install torch --index-url https://download.pytorch.org/whl/cu132
# If Newton v1.0.0 rejects this torch version, fall back to an older cu128 build (e.g. torch 2.7-2.9).

# Newton: https://newton-physics.github.io/newton/guide/installation.html
# MimicKit was tested with Newton v1.0.0. Newton 1.0.0 does not pin Warp, and the newest Warp (1.18)
# breaks it at import ("module 'warp.config' has no attribute 'verbose'"), so pin Warp to 1.12.1.
# The [sim] extra installs the MuJoCo backend that MimicKit's Newton engine uses as its solver.
python -m pip install "newton[sim]==1.0.0" "warp-lang==1.12.1"
python -m pip install trimesh scipy # trimesh loads the Go2 meshes; scipy silences its "unable to use sparse matrix" warnings

cd C:\Users\hgtll\Documents\Repos\masters\MimicKit
pip install -r requirements.txt
```
This exact combination was verified on this machine (Windows 11, RTX 5060 Ti): torch 2.15.0+cu132,
newton 1.0.0, warp-lang 1.12.1, mujoco / mujoco-warp 3.5.0 / 3.5.0.2, numpy 2.4.6. A 5-minute headless AMP Go2 run
with `--num_envs 256` trained without errors (about 150 iterations).
Engine flag: `--engine_config data/engines/newton_engine.yaml`.

> I haven't installed or run Newton here, so check its install page for exact commands, and expect to read
> `data/engines/newton_engine.yaml` once. The Go2 env config has a comment
> (`ground_contact_height: 0.15 # this is needed for IsaacGym`) showing some values are engine-specific.
> If Go2 behaves oddly under Newton, that block is the first place to look.

Quick GPU check for either path:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 3. Preview the Go2 motion data

Before training, confirm the sim + assets + motion clips load. The `view_motion` env just plays the reference clip:

```bash
cd MimicKit
python mimickit/run.py --mode test --arg_file args/view_motion_go2_args.txt --visualize true
```
(`args/view_motion_go2_args.txt` is hard-wired to Isaac Gym. Under Newton, add
`--engine_config data/engines/newton_engine.yaml` on the command line to override it.)

Viewer controls: `Alt` + left-drag pans, scroll zooms, `Enter` pauses, `Space` single-steps.
The reference clip is `data/motions/go2/go2_pace.pkl` (see `motion_file` in `data/envs/view_motion_go2_env.yaml`).

---

## 4. What the AMP Go2 config does

`args/amp_go2_args.txt` bundles three configs:

- **Env** `data/envs/amp_go2_env.yaml`: `go2.xml`, 10 s episodes, `motion_file: data/motions/go2/go2_pace.pkl`, key/contact
  bodies are the 4 feet, discriminator sees 10 observation steps (`num_disc_obs_steps`).
- **Agent** `data/agents/amp_go2_agent.yaml`: PPO actor/critic + a discriminator (all 2x1024 MLPs).
  `task_reward_weight: 0.0` and `disc_reward_weight: 1.0` means the policy is rewarded **only** for fooling the
  discriminator, i.e. looking like the motion data. No task goal.
- **Engine**: control at 30 Hz, simulation at 120 Hz.

So by default you train Go2 to reproduce a **pace** gait. To learn from several gaits at once, point `motion_file` at the
dataset file, which lists pace, trot, walk0-3 and run (run weighted 5x):

```yaml
# data/envs/amp_go2_env.yaml
motion_file: "data/datasets/dataset_go2_locomotion.yaml"
```
Note: with the single-clip default there is no commanded velocity or direction. The robot just imitates the clip's style.

---

## 5. Train

From `MimicKit/`, with the env activated:

```bash
# Isaac Gym (Path A)
python mimickit/run.py --arg_file args/amp_go2_args.txt --visualize false --logger tb

# Newton (Path B)
python mimickit/run.py --arg_file args/amp_go2_args.txt --engine_config data/engines/newton_engine.yaml --visualize false --logger tb
```

Notes:
- `--visualize false` for speed. Use `true` with a small `--num_envs 16` just to watch it start.
- `--num_envs 4096` is the default. A 16 GB card should handle it for Go2; if you get out-of-memory errors, try 2048 or 1024.
- If a flag is given both in the arg file and the command line, the command line is used (the README says arg-file
  arguments are "treated the same as command line arguments"). If a duplicate flag misbehaves, edit the arg file instead.
- `--logger` options: `txt`, `tb` (TensorBoard), `wandb`. `--out_dir` defaults to `output/` via the arg file.
- Checkpoints are saved every `iters_per_output: 100` iterations into the output directory.
- Stop with `Ctrl+C` once the reward/behavior plateaus. Training does not stop on its own at a convergence criterion that
  I could find in the config; watch the curves.

Monitor in a second terminal:
```bash
tensorboard --logdir=output/ --port=6006 --samples_per_plugin scalars=999999
```
Open http://localhost:6006. For `--logger txt`, plot `log.txt` with `tools/plot_log/plot_log.py`.

Distributed training (if you ever have more GPUs): add `--devices cuda:0 cuda:1`.

---

## 6. Test a trained model

```bash
python mimickit/run.py --arg_file args/amp_go2_args.txt --mode test --num_envs 4 --visualize true --model_file output/model.pt
```
Point `--model_file` at the actual checkpoint file produced in your `output/` folder (check the filename there).
Add the same `--engine_config` override as in step 5 if you use Newton.

What good looks like: the Go2 maintains the paced gait (legs on each side moving together) without falling, and
`pose_termination` is off, so episodes only end early on body-contact terminations (`enable_early_termination: True`).

---

## 7. Common problems

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: isaacgym` | Isaac Gym not installed in the active env; `pip install -e isaacgym/python` (Path A). |
| `AttributeError: module 'numpy' has no attribute 'float'` | numpy too new; `pip install numpy==1.23.5` (Path A). |
| `no kernel image is available for execution on the device` | Your torch wheel doesn't include your GPU's architecture. This is the Blackwell + old torch problem from section 0. |
| `FileNotFoundError: data/assets/go2/go2.xml` or `.../go2_pace.pkl` | Step 1 not done, or you ran from the wrong directory. Run from `MimicKit/`. |
| Out of GPU memory | Lower `--num_envs`. |
| Robot jitters or sinks into the ground on a new engine | Engine-specific `ground_contact_height` in the env/engine yaml; compare Isaac Gym vs Newton values. |
| Policy ends up standing still or falling at the start | Let it train longer first; AMP can take a while to leave early local minima. Then try the full locomotion dataset (section 4). |

---

## 8. Where to go next

- Add a task (e.g. follow a commanded velocity) the way `amp_steering_humanoid_*` does for the humanoid; that uses
  `amp_task_humanoid_agent.yaml` with a non-zero `task_reward_weight`. There is no Go2 version, so you would need to
  write a Go2 steering env.
- Compare with DeepMimic on the same clip: `args/deepmimic_go2_ppo_args.txt` (explicit pose tracking instead of a learned discriminator).
- Bring your own Go2 motion: `tools/gmr_to_mimickit/` converts GMR retargeted data to the `.pkl` format.

---

## 9. Makefile shortcuts

`MimicKit/Makefile` wraps the commands above. GNU make is not installed by default on Windows; it is installed
in the `mimickit` conda env (`conda install -n mimickit -c conda-forge make`). From `MimicKit/` with the env active:

```powershell
make              # list targets
make check        # torch + Warp see the GPU?
make view         # play the Go2 reference motion
make train        # AMP training, headless, 4096 envs -> output/go2_amp
make train-quick  # ~6 min smoke test
make train-view   # train with the viewer (16 envs)
make resume       # continue from output/go2_amp/model.pt
make test         # run the trained model in the viewer
make tensorboard  # http://localhost:6006
```
Override variables per call: `make train NUM_ENVS=2048 OUT=output/run2`, `make test MODEL=output/run2/model.pt`,
`make train ENGINE=isaac_gym`. Defaults are at the top of the Makefile. `make tensorboard` needs the `tensorboard`
package (installed in your env; `requirements.txt` only lists `tensorboardX`).
