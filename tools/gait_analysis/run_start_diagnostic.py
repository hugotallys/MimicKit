"""
Diagnostic 1: start every robot inside the run clip and watch which gait it keeps.

Question: is "run" a mode the policy can perform and leaves, or one it cannot sustain?

Every robot is reset at a random frame of the run clip (pose and velocity of the MoCap, ~3.2 m/s).
The steering command is then held fixed for the whole rollout:
  speed      --cmd_speeds (default 3.0 m/s, the run speed, and 1.1 m/s, the walk speed, as a control)
  direction  the robot's initial heading (it is already running that way)
so leaving the run cannot be explained by the task asking for something slower.

Only the first episode of each robot is used (no resets); a robot that falls is counted as fallen
from then on. Gaits are labelled with the same footfall classifier as gait_collapse.py, on 1-s
windows that end at each time t (stride --stride_steps).

Nothing in MimicKit is modified: the env's clip sampler and steering target are overridden on the
env instance inside this script only.

Outputs in <run_dir>/mode_diagnostics/:
  start_from_run.csv   per command speed and window end time: fraction run / trot / walk_pace / stand /
                       fallen, mean speed along the target direction
  start_from_run.txt   summary: time until each robot leaves the run, gait it ends in, fall rate
  start_from_run.png   gait fractions over time, one panel per command speed

Usage (from MimicKit/):
  python tools/gait_analysis/run_start_diagnostic.py --run_dir output/go2_steer_amp_allgaits_2048env_s42
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "mimickit"))

import gait_collapse as gc  # noqa: E402  (also puts mimickit/ on the path)
import envs.base_env as base_env  # noqa: E402
import envs.env_builder as env_builder  # noqa: E402
import learning.agent_builder as agent_builder  # noqa: E402
import learning.base_agent as base_agent  # noqa: E402
import util.mp_util as mp_util  # noqa: E402
import util.torch_util as torch_util  # noqa: E402
import util.util as util  # noqa: E402

RUN_CLIP = "go2_run"


def find_clip(env, name):
    ml = env._motion_lib
    for m in range(ml.get_num_motions()):
        if os.path.splitext(os.path.basename(ml.get_motion_file(m)))[0] == name:
            return m
    raise ValueError("clip {} not in the dataset".format(name))


def force_clip(env, clip_id):
    """Make every reset start at a random frame of one clip (overrides the instance method only)."""
    ml, dev = env._motion_lib, env._device

    def sample_motion_times(n):
        ids = torch.full([n], clip_id, device=dev, dtype=torch.long)
        return ids, ml.sample_time(ids)

    env._sample_motion_times = sample_motion_times


def hold_command(env, speed):
    """Fixed target: given speed, along each robot's current heading; never resampled."""
    root_rot = env._engine.get_root_rot(env._get_char_id())
    h = torch_util.calc_heading(root_rot)
    d = torch.stack([torch.cos(h), torch.sin(h)], dim=-1)
    env._tar_speed[:] = speed
    env._tar_dir[:] = d
    env._face_dir[:] = d
    env._tar_change_times[:] = 1e9
    env._update_observations()       # the first action already sees the new command
    return env._obs_buf, env._info


def rollout(env, agent, speed, num_steps):
    char_id, foot_ids = env._get_char_id(), env._key_body_ids
    n, dev, e = env.get_num_envs(), env._device, env._engine

    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)
    env.reset()
    obs, info = hold_command(env, speed)
    tar_dir = env._tar_dir.clone()

    hist = None
    v_along = torch.zeros([n, num_steps], device=dev)
    end_step = torch.full([n], num_steps, device=dev, dtype=torch.long)   # first step after the episode ended
    fell = torch.zeros(n, device=dev, dtype=torch.bool)
    with torch.no_grad():
        for t in range(num_steps):
            action, _ = agent._decide_action(obs, info)
            obs, _, done, info = env.step(action)
            f = gc.step_features(e.get_root_pos(char_id), e.get_root_rot(char_id), e.get_root_vel(char_id),
                                 e.get_dof_pos(char_id), e.get_body_pos(char_id)[:, foot_ids, :])
            if hist is None:
                hist = torch.zeros([n, num_steps, f.shape[-1]], device=dev)
            hist[:, t] = f
            v_along[:, t] = (e.get_root_vel(char_id)[:, :2] * tar_dir).sum(-1)

            new_done = (done != base_env.DoneFlags.NULL.value) & (end_step == num_steps)
            fell |= new_done & (done == base_env.DoneFlags.FAIL.value)
            end_step[new_done] = t
    return hist, v_along, end_step, fell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--model", default=None, help="checkpoint (default: <run_dir>/model.pt)")
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--rollout_sec", type=float, default=10.0)
    ap.add_argument("--win_sec", type=float, default=1.0)
    ap.add_argument("--stride_steps", type=int, default=3)
    ap.add_argument("--cmd_speeds", default="3.0,1.1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = os.path.join(args.run_dir, "mode_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    model = args.model or os.path.join(args.run_dir, "model.pt")

    mp_util.init(0, 1, args.device, None)
    util.set_rand_seed(args.seed)
    env = env_builder.build_env(os.path.join(args.run_dir, "env_config.yaml"),
                                os.path.join(args.run_dir, "engine_config.yaml"),
                                args.num_envs, args.device, visualize=False)
    agent = agent_builder.build_agent(os.path.join(args.run_dir, "agent_config.yaml"), env, args.device)
    agent.load(model)

    run_id = find_clip(env, RUN_CLIP)
    force_clip(env, run_id)
    dt = env._engine.get_timestep()
    nd = env._engine.get_dof_pos(env._get_char_id()).shape[-1]
    W = int(round(args.win_sec / dt))
    steps = int(round(args.rollout_sec / dt))
    ends = np.arange(W - 1, steps, args.stride_steps)
    i_run = gc.GAITS.index("run")

    rows, lines, results = [], [], {}
    lines.append("Diagnostic 1: every robot starts inside the run clip ({}), {} robots, {:.0f} s, model {}".format(
        RUN_CLIP, args.num_envs, args.rollout_sec, model))
    lines.append("Command held fixed: speed below, direction = initial heading. Windows of {:.1f} s.".format(args.win_sec))
    lines.append("")

    for speed in [float(s) for s in args.cmd_speeds.split(",")]:
        util.set_rand_seed(args.seed)
        hist, v_along, end_step, fell = rollout(env, agent, speed, steps)

        offs = torch.arange(-W + 1, 1, device=hist.device)
        win = hist[:, torch.as_tensor(ends, device=hist.device)[:, None] + offs[None, :]]   # [n, T, W, F]
        n, T = win.shape[0], win.shape[1]
        flat = win.reshape(n * T, W, -1)
        st = gc.footfall_stats(flat, nd, gc.resting_foot_height(flat, nd))
        lab = gc.footfall_classify(st).reshape(n, T).cpu().numpy()
        end_np, fell_np = end_step.cpu().numpy(), fell.cpu().numpy()
        valid = ends[None, :] < end_np[:, None]            # window fully inside the episode
        fallen = (~valid) & fell_np[:, None]
        va = v_along.cpu().numpy()

        for k, t_end in enumerate(ends):
            v = valid[:, k]
            row = {"cmd_speed": speed, "t": (t_end + 1) * dt}
            for i, g in enumerate(gc.GAITS):
                row["p_" + g] = float(((lab[:, k] == i) & v).mean())
            row["p_fallen"] = float(fallen[:, k].mean())
            row["v_along"] = float(va[v, max(t_end - W + 1, 0):t_end + 1].mean()) if v.any() else float("nan")
            rows.append(row)

        # time until a robot leaves the run: first window not labelled run (or the fall)
        not_run = (lab != i_run) | ~valid
        first = np.where(not_run.any(1), not_run.argmax(1), T)
        t_leave = np.where(first < T, (ends[np.minimum(first, T - 1)] + 1) * dt, np.inf)
        started_run = valid[:, 0] & (lab[:, 0] == i_run)
        alive_end = valid[:, -1]
        final_gait = np.bincount(lab[alive_end, -1], minlength=len(gc.GAITS)) / max(alive_end.sum(), 1)
        results[speed] = (rows[-T:], t_leave, started_run)

        tl = t_leave[started_run]
        lines.append("Command {:.1f} m/s".format(speed))
        lines.append("  first window (0-{:.1f} s) labelled run: {:.0%} of robots".format(args.win_sec, started_run.mean()))
        if len(tl):
            stay = np.isinf(tl)
            lines.append("  among those: left the run after median {:.2f} s (p25 {:.2f}, p75 {:.2f}); still running at the end: {:.0%}".format(
                np.median(tl[~stay]) if (~stay).any() else float("nan"),
                np.percentile(tl[~stay], 25) if (~stay).any() else float("nan"),
                np.percentile(tl[~stay], 75) if (~stay).any() else float("nan"), stay.mean()))
        lines.append("  fell during the {:.0f} s: {:.0%}".format(args.rollout_sec, fell_np.mean()))
        if fell_np.any():
            ft = (end_np[fell_np] + 1) * dt
            lines.append("  fall time: median {:.2f} s, p90 {:.2f} s, {:.0%} of the falls before 1 s".format(
                np.median(ft), np.percentile(ft, 90), (ft < 1.0).mean()))

        # step resolution over the first 1.5 s: speed and flight phase (all four feet off the ground),
        # robots still in their episode only; the MoCap run clip has ~15% flight
        base = gc.resting_foot_height(flat, nd)
        feet = hist[..., nd + 4:]
        flight = ((feet - base) >= gc.CONTACT_THR).all(-1).float().cpu().numpy()      # [n, steps]
        alive_step = np.arange(steps)[None, :] < end_np[:, None]
        lines.append("  step resolution (robots still up): time | speed along target | flight phase")
        for t_s in [0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
            k = int(round(t_s / dt)) - 1
            m = alive_step[:, k]
            # flight over a 0.2 s span ending at k (a single step is too noisy)
            k0 = max(0, k - int(round(0.2 / dt)) + 1)
            lines.append("    {:4.2f} s | {:.2f} m/s | {:.0%}".format(
                (k + 1) * dt, va[m, k].mean(), flight[m, k0:k + 1].mean()))
        lines.append("  gait at the end (robots still up): " +
                     ", ".join("{}={:.0%}".format(g, p) for g, p in zip(gc.GAITS, final_gait)))
        for t_check in [1.0, 2.0, 3.0, 5.0, args.rollout_sec]:
            k = int(np.argmin(np.abs((ends + 1) * dt - t_check)))
            r = rows[-T + k]
            lines.append("  t={:4.1f} s: run={:.0%} trot={:.0%} walk_pace={:.0%} stand={:.0%} fallen={:.0%} v={:.2f} m/s".format(
                r["t"], r["p_run"], r["p_trot"], r["p_walk_pace"], r["p_stand"], r["p_fallen"], r["v_along"]))
        lines.append("")

    with open(os.path.join(out_dir, "start_from_run.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, "start_from_run.txt"), "w") as f:
        f.write(report + "\n")

    # ---------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_gait_collapse as pg

    speeds = list(results)
    fig, axs = plt.subplots(1, len(speeds), figsize=(5.2 * len(speeds), 3.4), sharey=True, squeeze=False)
    for ax, speed in zip(axs[0], speeds):
        rs = results[speed][0]
        t = np.array([r["t"] for r in rs])
        order = ["run", "trot", "walk_pace", "stand"]
        parts = [np.array([r["p_" + g] for r in rs]) for g in order] + [np.array([r["p_fallen"] for r in rs])]
        cols = [pg.GAIT_COLORS[g] for g in order] + [pg.OFF_COLOR]
        labs = [pg.LABELS[g] for g in order] + ["fallen"]
        ax.stackplot(t, *parts, colors=cols, labels=labs, edgecolor="white", linewidth=0.4)
        ax2 = ax.twinx()
        ax2.plot(t, [r["v_along"] for r in rs], color=pg.INK, lw=1.2)
        ax2.axhline(speed, color=pg.INK2, lw=0.8, ls="--")
        ax2.text(t[-1], speed, "commanded", color=pg.INK2, fontsize=8, ha="right", va="bottom")
        ax2.set_ylim(0, 4)
        ax2.set_ylabel("speed along target (m/s)", color=pg.INK)
        ax2.grid(False)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel("time since start (s, end of 1-s window)")
        ax.set_title("Start in run clip, command {:.1f} m/s".format(speed), loc="left", color=pg.INK)
    axs[0][0].set_ylabel("fraction of robots")
    axs[0][0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "start_from_run.png"), bbox_inches="tight")
    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
