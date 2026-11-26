"""
Footfall (Hildebrand-style) diagram of the reference clips: one row per foot, a bar where the foot is
on the ground, for walk, pace, trot and run. Diagonal pairs share a colour, so a trot shows aligned
bars of the same colour, a pace aligned bars of different colours, and a run gaps where no foot is
down (flight, shaded). Each panel reports the three numbers the gait classifier uses on that window.

Contact uses the same rule as gait_collapse.py: foot height less than CONTACT_THR above the resting
foot height (2nd percentile of all reference foot heights).

Usage (from MimicKit/):
  python tools/gait_analysis/plot_footfall_diagram.py --run_dir output/go2_steer_amp_allgaits_2048env_s42 \
      --out output/figures/footfall_diagram.png
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "mimickit"))

import gait_collapse as gc  # noqa: E402
import envs.env_builder as env_builder  # noqa: E402
import util.mp_util as mp_util  # noqa: E402

CLIPS = [("go2_walk1", "walk"), ("go2_pace", "pace"), ("go2_trot", "trot"), ("go2_run", "run")]
FEET = ["FR", "FL", "RR", "RL"]          # key_bodies order in the env config
ROWS = [("FR", 0), ("RL", 3), ("FL", 1), ("RR", 2)]   # diagonal pairs next to each other
DIAG_A, DIAG_B = "#2a6fc4", "#d4572a"   # FR+RL, FL+RR


def foot_heights(env, clip_id, t0, n, dt):
    ml = env._motion_lib
    times = torch.tensor(t0 + dt * np.arange(n), device=env._device, dtype=torch.float)
    ids = torch.full_like(times, clip_id, dtype=torch.long)
    root_pos, root_rot, root_vel, _, joint_rot, _ = ml.calc_motion_frame(ids, times)
    body_pos, _ = env._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
    speed = torch.linalg.norm(root_vel[:, :2], dim=-1)
    return body_pos[:, env._key_body_ids, 2].cpu().numpy(), speed.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="any run whose env_config holds the all-gaits dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--win_sec", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    mp_util.init(0, 1, args.device, None)
    env = env_builder.build_env(os.path.join(args.run_dir, "env_config.yaml"),
                                os.path.join(args.run_dir, "engine_config.yaml"),
                                1, args.device, visualize=False)
    ml, dt = env._motion_lib, env._engine.get_timestep()
    names = [os.path.splitext(os.path.basename(ml.get_motion_file(m)))[0] for m in range(ml.get_num_motions())]
    n = int(round(args.win_sec / dt))

    # resting foot height over all clips, as in gait_collapse.resting_foot_height
    all_h = np.concatenate([foot_heights(env, m, 0.0, int(ml.get_motion_lengths()[m].item() / dt) - 1, dt)[0].ravel()
                            for m in range(ml.get_num_motions())])
    base = np.percentile(all_h, 2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_gait_collapse as pg

    fig, axs = plt.subplots(1, 4, figsize=(16.0, 3.6), sharey=True)
    for ax, (clip, label) in zip(axs, CLIPS):
        m = names.index(clip)
        length = ml.get_motion_lengths()[m].item()
        t0 = max(0.0, 0.5 * (length - args.win_sec))           # window in the middle of the clip
        h, v = foot_heights(env, m, t0, n, dt)
        c = (h - base) < gc.CONTACT_THR                         # [n, 4] on the ground
        diag = 0.5 * ((c[:, 0] == c[:, 3]).mean() + (c[:, 1] == c[:, 2]).mean())
        lat = 0.5 * ((c[:, 0] == c[:, 2]).mean() + (c[:, 1] == c[:, 3]).mean())
        flight = (~c.any(1)).mean()
        t = np.arange(n) * dt

        for k in np.where(~c.any(1))[0]:                        # flight: no foot down
            ax.axvspan(t[k], t[k] + dt, color=pg.OFF_COLOR, alpha=0.45, lw=0)
        for row, (foot, j) in enumerate(ROWS):
            col = DIAG_A if foot in ("FR", "RL") else DIAG_B
            on = c[:, j]
            k = 0
            while k < n:
                if on[k]:
                    e = k
                    while e < n and on[e]:
                        e += 1
                    ax.add_patch(plt.Rectangle((t[k], row + 0.15), (e - k) * dt, 0.7, color=col, lw=0))
                    k = e
                else:
                    k += 1
        ax.set_xlim(0, n * dt)
        ax.set_ylim(-0.1, len(ROWS))
        ax.set_yticks([r + 0.5 for r in range(len(ROWS))])
        ax.set_yticklabels([f for f, _ in ROWS])
        ax.invert_yaxis()
        ax.grid(False)
        ax.set_xlabel("time (s)")
        ax.set_title("{}\ndiag − lat = {:+.2f} · flight {:.0%} · {:.1f} m/s".format(label, diag - lat, flight, v.mean()),
                     loc="left", color=pg.INK, fontsize=10)
        print("{:5s} diag={:.2f} lat={:.2f} diag-lat={:+.2f} flight={:.2f} speed={:.2f}".format(
            label, diag, lat, diag - lat, flight, v.mean()))

    handles = [plt.Rectangle((0, 0), 1, 1, color=DIAG_A), plt.Rectangle((0, 0), 1, 1, color=DIAG_B),
               plt.Rectangle((0, 0), 1, 1, color=pg.OFF_COLOR, alpha=0.45)]
    fig.legend(handles, ["diagonal pair FR + RL (foot on the ground)", "diagonal pair FL + RR (foot on the ground)",
                         "flight: no foot on the ground"], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
