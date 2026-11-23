"""
Test 3 figure: is running feasible for the Go2? Final checkpoint of an AMP run trained on the run clip
alone, against the MoCap run clip and the final multi-gait policy.

  (a) footfall signature: diagonal - lateral sync vs speed, one dot per 1-s window
  (b) flight phase (fraction of the window with all four feet in the air) per source

Inputs are the windows.npz files written by gait_collapse.py.

Usage (from MimicKit/):
  python tools/gait_analysis/plot_run_feasibility.py --run_only output/go2_amp_run_only_2048env_s42 \
      --multi output/go2_steer_amp_allgaits_2048env_s42 --out output/figures/gait_run_only_s42
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_gait_collapse as pg


def load(run):
    return np.load(os.path.join(run, "gait_analysis", "windows.npz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_only", required=True)
    ap.add_argument("--multi", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    r, m = load(args.run_only), load(args.multi)
    on_r, on_m = r["final_on"], m["final_on"]
    sources = [
        ("MoCap run clip", r["ref_speed"], r["ref_diag"] - r["ref_lat"], r["ref_flight"], pg.GAIT_COLORS["run"], 0.9),
        ("AMP, run clip only (iter {})".format(int(r["final_iter"])), r["final_speed"][on_r],
         (r["final_diag"] - r["final_lat"])[on_r], r["final_flight"][on_r], pg.GAIT_COLORS["run"], 0.25),
        ("AMP, all gaits + steering (iter {})".format(int(m["final_iter"])), m["final_speed"][on_m],
         (m["final_diag"] - m["final_lat"])[on_m], m["final_flight"][on_m], pg.OFF_COLOR, 0.25),
    ]

    fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.6), gridspec_kw={"width_ratios": [1.6, 1.0]})
    for name, v, s, _, c, a in sources[::-1]:
        axs[0].scatter(v, s, s=6 if a < 0.5 else 14, color=c, alpha=a, lw=0, label=name)
    axs[0].axvline(2.0, color=pg.INK2, lw=0.8, ls=":")
    axs[0].text(2.03, axs[0].get_ylim()[1], "run speed\nthreshold", color=pg.INK2, fontsize=8, va="top")
    axs[0].set_xlabel("root speed (m/s)")
    axs[0].set_ylabel("diagonal sync - lateral sync")
    axs[0].set_title("Footfall signature per 1-s window", loc="left", color=pg.INK)
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=3,
                  markerscale=2, fontsize=8)

    data = [f for _, _, _, f, _, _ in sources]
    bp = axs[1].boxplot(data, patch_artist=True, showfliers=False, widths=0.55)
    for patch, (_, _, _, _, c, a) in zip(bp["boxes"], sources):
        patch.set_facecolor(c)
        patch.set_alpha(0.85 if a > 0.5 else 0.45)
        patch.set_edgecolor(pg.INK2)
    for med in bp["medians"]:
        med.set_color(pg.INK)
    axs[1].set_xticks([1, 2, 3])
    axs[1].set_xticklabels(["MoCap\nrun", "AMP\nrun only", "AMP\nall gaits"], fontsize=8)
    axs[1].set_ylabel("flight phase (fraction of window)")
    axs[1].set_title("Flight phase", loc="left", color=pg.INK)
    axs[1].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "run_feasibility.png"), bbox_inches="tight")

    for name, v, s, f, _, _ in sources:
        print("{:<40} n={:6d} speed={:.2f} m/s  diag-lat={:+.2f}  flight={:.3f}".format(name, len(v), v.mean(), s.mean(), f.mean()))
    print("Saved to", args.out)


if __name__ == "__main__":
    main()
