"""
Reward curves of one or more runs as mean +- 1 std (solid line + shaded band), from log.txt.

  task reward   Test_Return +- Test_Return_Std: return of the task reward per test episode, std
                across the test episodes of that evaluation (runs logged before the std was added
                show the mean only)
  style reward  Disc_Reward_Mean +- Disc_Reward_Std: style (discriminator) reward per transition,
                std across the transitions of that iteration's batch

The std is the spread inside one run (between episodes / transitions), not across seeds.

Usage (from MimicKit/):
  python tools/gait_analysis/plot_reward_curves.py --out output/figures/reward_curves.png \
      --runs output/go2_steer_amp_allgaits_2048env_s42 output/go2_loc_amp_allgaits_s42 \
      --labels "all gaits, steering" "all gaits, location"
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_gait_collapse as pg

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"]


def gc_split_header(line):
    """Column names of MimicKit's txt log. Names are padded to 25 characters, so a longer name
    (Disc_Agent_Logit_Walk_Pace) runs into the next one: split where a lowercase letter meets an uppercase."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", line).split()


def read_log(run):
    with open(os.path.join(run, "log.txt"), newline="") as f:
        lines = [r for r in re.split(r"[\r\n]+", f.read()) if r.strip()]
    rows = [gc_split_header(lines[0])] + [r.split() for r in lines[1:]]
    h = rows[0]
    data = np.array([[float(x) for x in r] for r in rows[1:] if len(r) == len(h)])
    return {k: data[:, i] for i, k in enumerate(h)}


def band(ax, x, m, s, color, label):
    ax.plot(x, m, color=color, lw=2.0, label=label)
    if s is not None:
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.2, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--no_task", nargs="*", default=[], help="labels of runs without a task reward (skipped in the task panel)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.6))
    for i, (run, label) in enumerate(zip(args.runs, args.labels)):
        d = read_log(run)
        x = d["Samples"] / 1e6
        c = COLORS[i % len(COLORS)]
        if label not in args.no_task:
            band(axs[0], x, d["Test_Return"], d.get("Test_Return_Std"), c,
                 label + ("" if "Test_Return_Std" in d else " (mean only)"))
        band(axs[1], x, d["Disc_Reward_Mean"], d["Disc_Reward_Std"], c, label)
        print("{:<30} final: task {:.1f} +- {:.1f}, style {:.2f} +- {:.2f}".format(
            label, d["Test_Return"][-1], d.get("Test_Return_Std", [np.nan])[-1],
            d["Disc_Reward_Mean"][-1], d["Disc_Reward_Std"][-1]))

    axs[0].set_title("Task reward per test episode (mean ± 1 std)", loc="left", color=pg.INK)
    axs[0].set_ylabel("episode return")
    axs[1].set_title("Style reward per transition (mean ± 1 std)", loc="left", color=pg.INK)
    axs[1].set_ylabel("discriminator reward")
    # the untrained discriminator at iteration 0 gives a spike that hides the rest of the curve
    axs[1].set_ylim(-0.3, 1.6)
    for a, loc in zip(axs, ["upper left", "lower right"]):
        a.set_xlabel("training samples (M)")
        a.set_xlim(left=0)
        a.legend(loc=loc, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
