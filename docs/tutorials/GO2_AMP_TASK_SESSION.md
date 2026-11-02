# Go2 AMP + target-location task: session notes

Setup and results of the first AMP + task training on the Unitree Go2 with MimicKit (Newton engine, Windows 11,
RTX 5060 Ti). General setup steps are in `GO2_AMP_TUTORIAL.md`.

## Quick commands

Run from `MimicKit/` with the conda env active (`conda activate mimickit`).

Evaluate the final model in the viewer (1 env; use `--num_envs 256` for a crowd):

```bash
python mimickit/run.py --arg_file args/amp_location_go2_args.txt --engine_config data/engines/newton_engine.yaml --mode test --num_envs 1 --visualize true --model_file output/go2_loc_amp_final/model.pt
```

Train again from scratch (new output folder so nothing is overwritten):

```bash
python mimickit/run.py --arg_file args/amp_location_go2_args.txt --engine_config data/engines/newton_engine.yaml --num_envs 4096 --visualize false --logger tb --out_dir output/go2_loc_amp_new
```

Resume from the final model (optimizer state, counters and the discriminator replay buffer are not restored):

```bash
python mimickit/run.py --arg_file args/amp_location_go2_args.txt --engine_config data/engines/newton_engine.yaml --num_envs 4096 --visualize false --logger tb --model_file output/go2_loc_amp_final/model.pt --out_dir output/go2_loc_amp_b
```

Note: `make test`, `make resume` and `make train` use the plain Go2 AMP args (`amp_go2_args.txt`), so for this task
model use the full commands above.

Viewer controls: `Alt` + left-drag pans, scroll zooms, `Enter` pauses, `Space` single-steps. Close the window or press
`Ctrl+C` to quit.

## TensorBoard

One run:

```bash
python -m tensorboard.main --logdir=output/go2_loc_amp_final --port=6006 --samples_per_plugin scalars=999999
```

Same thing through make (set `OUT`):

```bash
make tensorboard OUT=output/go2_loc_amp_final
```

Compare several runs: point `--logdir` at the parent folder (`--logdir=output`); each run appears in the "Runs" list.
Then open http://localhost:6006.

- Points are written every `iters_per_output` (100) iterations, and the writer buffers them, so expect a delay of
  a couple of minutes before a new point shows. Press the refresh icon (top bar) after waiting.
- The dark line is exponentially smoothed (slider in Settings, default 0.6); the faint line is the raw data.
- Useful curves: `0_Main/Test_Return`, `0_Main/Train_Return`, `Train_Episode_Length`, `Misc/Disc_Reward_Mean`,
  `Misc/Critic_Loss`, `Misc/Clip_Frac`.

## Setup done

- Environment `mimickit`: Python 3.11, torch 2.15.0+cu132, newton 1.0.0 (`newton[sim]`), warp-lang **1.12.1**
  (1.18 breaks Newton 1.0.0 at import), mujoco / mujoco-warp 3.5.0 / 3.5.0.2, trimesh, scipy, tensorboard.
- Isaac Gym was not used: it is Linux-only with Python 3.6-3.8 bindings, which cannot work with a Blackwell GPU.
- Data from `MimicKit_Data` merged into `MimicKit/data/`.
- Files added: `Makefile`, `data/envs/amp_location_go2_env.yaml`, `data/agents/amp_task_go2_agent.yaml`,
  `args/amp_location_go2_args.txt`.
- Task env: `env_name: "task_location"` (`TaskLocationEnv`), `episode_length: 20.0`, `global_obs: False`,
  `tar_speed: 1.0`, target re-sampled every 5-10 s within 10 m. Style clip: `data/motions/go2/go2_pace.pkl`.
- Agent: AMP, `task_reward_weight: 0.5`, `disc_reward_weight: 0.5`, `action_std: 0.1`, 4096 envs.

## Training runs

| Run folder | Setup | Final iter | Samples | Hours | Notes |
|---|---|---|---|---|---|
| `output/go2_amp` | AMP, pace clip, style only | 500 | 66M | 0.64 | Stopped early as a test. Train ep. length about 295/300, `Disc_Reward_Mean` 0.52. |
| `output/go2_run` | AMP, run clip, style only | 1000 | 131M | 3.12 | Train ep. length about 266/300, `Disc_Reward_Mean` 0.59, still rising when stopped. |
| `output/go2_loc_amp_final` | AMP + location task (0.5 / 0.5), pace clip | 2500 | 328M | 4.21 | See below. |

`output/go2_loc_amp_1900` and `output/go2_loc_amp_2300` are snapshots of the task run at those iterations.
For the style-only runs `Test_Return` and `Train_Return` stay at 0 because `task_reward_weight` is 0.

### Task run metrics (final model, iteration 2500)

| Metric | Value |
|---|---|
| Test return | 381 (max possible is about 600: 20 s x 30 Hz x about 1.0 per step) |
| Train return | 369 |
| Test / train episode length | 597 / 589 of 600 steps (the robot does not fall) |
| `Disc_Reward_Mean` | 0.51 |
| `Disc_Agent_Acc` | 0.997 |
| `Critic_Loss` | 8.1 |
| `Clip_Frac` | 0.116 |

Trajectory: task return sat at about 80 for the first 100M samples, then took off to about 350 by 220M, and slowly
crept up after that (test return is noisy; it is a 32-episode average). `Disc_Reward_Mean` dipped to 0.26 at about
80M, then rose to about 0.5 and flattened. Evaluation in the viewer looked good (confirmed by the user).

## Notes and caveats

- `Disc_Reward_Mean` is `2 * softplus(discriminator logit)`: about 1.39 would mean the discriminator cannot tell
  the policy from the demo, and it is not capped at 1. Around 0.5-0.9 is normal. The AMP discriminator accuracy
  near 1.0 is also normal.
- Actor loss is noisy around zero by design (normalized advantages) and is not a useful progress signal.
- A `opt.ccd_iterations ... needs to be increased` warning from MuJoCo can appear. It is harmless unless the feet
  visibly jitter; `ccd_iterations` is a `SolverMuJoCo` argument (`mimickit/engines/newton_engine.py`).
- Not done yet: the task-only baseline (no style reward) for comparison; I have not checked whether `Train_Return`
  logs the raw or the weighted task reward, so confirm that before comparing runs.
