"""
Figures for the gait mode-collapse analysis (output of gait_collapse.py).

One or more run directories (seeds of the same config) are combined: curves show the mean
across seeds with a +-1 std band when more than one seed is given.

  Fig. 1  gait_distribution.png  stacked area of p(gait) over training samples (mean over seeds)
  Fig. 2  collapse_metrics.png   normalized entropy, JS divergence to the reference, test return
  Fig. 3  footfall_final.png     footfall signature (diagonal - lateral sync vs speed),
                                 reference clips vs final policy (first run)
  Fig. 4  gait_by_speed.png      steering runs only (needs speed_bins.csv): gait per commanded-speed
                                 bin at the final checkpoint, commanded vs achieved speed, and the
                                 mutual information I(command; gait) over training

Usage (from MimicKit/):
  python tools/gait_analysis/plot_gait_collapse.py --runs output/go2_loc_amp_allgaits_s42 [more seeds...] \
      --out output/figures/gait_collapse
"""

import argparse
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# categorical slots 1-3 of the reference palette (fixed order), neutral gray for "off-manifold"
GAIT_COLORS = {"walk_pace": "#2a78d6", "trot": "#eb6834", "run": "#1baf7a", "stand": "#eda100", "hop": "#e87ba4"}
OFF_COLOR = "#b4b2a9"
INK, INK2, GRID = "#1f1e1d", "#5f5e5a", "#e6e4dd"
LABELS = {"walk_pace": "walk / pace", "trot": "trot", "run": "run",
          "stand": "stand (not in reference)", "hop": "hop (not in reference)"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300,
})


def read_metrics(run):
    with open(os.path.join(run, "gait_analysis", "metrics.csv")) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def read_log(run, key):
    with open(os.path.join(run, "log.txt"), newline="") as f:
        lines = [r for r in re.split(r"[\r\n]+", f.read()) if r.strip()]
    # names are padded to 25 characters; a longer one (Disc_Agent_Logit_Walk_Pace) runs into the next
    h = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", lines[0]).split()
    rows = [r.split() for r in lines[1:]]
    data = np.array([[float(x) for x in r] for r in rows if len(r) == len(h)])
    return data[:, h.index("Samples")], data[:, h.index(key)]


def align(series_x, series_y):
    """Truncate all seeds to the common checkpoints (same iteration grid)."""
    n = min(len(x) for x in series_x)
    return series_x[0][:n], np.stack([y[:n] for y in series_y])


def band(ax, x, ys, color, label):
    m = np.nanmean(ys, 0)
    ax.plot(x, m, color=color, lw=2.0, label=label)
    if ys.shape[0] > 1:
        s = np.nanstd(ys, 0)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.25, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="output/figures/gait_collapse")
    ap.add_argument("--title", default="AMP + location, all gaits")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    mets = [read_metrics(r) for r in args.runs]
    gaits = [k[2:] for k in mets[0] if k.startswith("p_")]
    x, _ = align([m["samples"] for m in mets], [m["samples"] for m in mets])
    xs = x / 1e6
    n_seeds = len(args.runs)
    seed_txt = "mean of {} seeds".format(n_seeds) if n_seeds > 1 else "1 seed"

    # ---------------- Fig. 1: gait distribution over training
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    on = align([m["samples"] for m in mets], [1 - m["off_manifold"] for m in mets])[1].mean(0)
    parts = [align([m["samples"] for m in mets], [np.nan_to_num(m["p_" + g]) for m in mets])[1].mean(0) * on
             for g in gaits]
    parts.append(1 - on)
    colors = [GAIT_COLORS[g] for g in gaits] + [OFF_COLOR]
    labels = [LABELS[g] for g in gaits] + ["off-manifold (falling / not in reference)"]
    ax.stackplot(xs, *parts, colors=colors, labels=labels, edgecolor="white", linewidth=0.6)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("training samples (M)")
    ax.set_ylabel("fraction of policy windows")
    ax.set_title("Gait used by the policy during training ({})".format(seed_txt), loc="left", color=INK)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "gait_distribution.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------------- Fig. 2: collapse metrics (three panels, one y-scale each)
    fig, axs = plt.subplots(1, 3, figsize=(10.0, 2.8))
    _, H = align([m["samples"] for m in mets], [m["H_norm"] for m in mets])
    _, JS = align([m["samples"] for m in mets], [m["JS"] for m in mets])
    q = np.load(os.path.join(args.runs[0], "gait_analysis", "windows.npz"))["ref_gait_dist"]
    qr = q[q > 0]
    ref_H = -(qr * np.log(qr)).sum() / np.log(len(qr))
    mid = 0.5 * (np.eye(len(q))[0] + q)            # policy = 100% walk/pace
    js_collapse = 0.5 * np.log2(1 / mid[0]) + 0.5 * (qr * np.log2(qr / mid[q > 0])).sum()

    band(axs[0], xs, H, GAIT_COLORS["walk_pace"], "policy")
    axs[0].axhline(ref_H, color=INK2, lw=1, ls="--")
    axs[0].text(xs[-1], ref_H, "reference", color=INK2, ha="right", va="bottom", fontsize=8)
    axs[0].set_ylim(0, 1)
    axs[0].set_title("Entropy over reference gaits", loc="left", color=INK)
    axs[0].set_xlabel("training samples (M)")

    band(axs[1], xs, JS, GAIT_COLORS["walk_pace"], "policy")
    axs[1].axhline(js_collapse, color=INK2, lw=1, ls="--")
    axs[1].text(xs[-1], js_collapse, "full collapse to walk/pace", color=INK2, ha="right", va="bottom", fontsize=8)
    axs[1].set_ylim(0, 0.6)
    axs[1].set_title("JS divergence to reference", loc="left", color=INK)
    axs[1].set_xlabel("training samples (M)")

    rets = [read_log(r, "Test_Return") for r in args.runs]
    rx, ry = align([r[0] for r in rets], [r[1] for r in rets])
    band(axs[2], rx / 1e6, ry, GAIT_COLORS["walk_pace"], "test return")
    axs[2].set_title("Test return", loc="left", color=INK)
    axs[2].set_xlabel("training samples (M)")
    for a in axs:
        a.set_xlim(0, xs[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "collapse_metrics.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------------- Fig. 3: footfall signature, reference vs final policy
    d = np.load(os.path.join(args.runs[0], "gait_analysis", "windows.npz"))
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.2), sharex=True, sharey=True)
    for i, g in enumerate(gaits):
        m = d["ref_labels"] == i
        if not m.any():
            continue
        axs[0].scatter(d["ref_speed"][m], d["ref_diag"][m] - d["ref_lat"][m], s=10, color=GAIT_COLORS[g],
                       alpha=0.6, lw=0, label=LABELS[g])
    on = d["final_on"]
    for i, g in enumerate(gaits):
        m = on & (d["final_ff"] == i)
        if not m.any():
            continue
        axs[1].scatter(d["final_speed"][m], d["final_diag"][m] - d["final_lat"][m], s=6, color=GAIT_COLORS[g],
                       alpha=0.35, lw=0, label="{} ({:.0%})".format(LABELS[g], m.sum() / max(on.sum(), 1)))
    for a, t in zip(axs, ["Reference MoCap (colour = clip label)", "Policy, final checkpoint (iter {})".format(int(d["final_iter"]))]):
        a.axhline(0.3, color=INK2, lw=0.8, ls=":")
        a.axvline(2.5, color=INK2, lw=0.8, ls=":")
        a.set_title(t, loc="left", color=INK)
        a.set_xlabel("root speed (m/s)")
        a.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2, markerscale=2, fontsize=8)
    axs[0].set_ylabel("diagonal sync - lateral sync")
    axs[0].text(0.05, 0.32, "trot region", color=INK2, fontsize=8)
    axs[0].annotate("trot segments inside\nthe walk clips", xy=(1.55, 0.7), xytext=(0.2, 0.55), color=INK2, fontsize=8,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    axs[0].text(2.55, 0.6, "run region", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "footfall_final.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------------- Fig. 4: gait vs commanded speed (steering only)
    sb_path = os.path.join(args.runs[0], "gait_analysis", "speed_bins.csv")
    if os.path.exists(sb_path):
        with open(sb_path) as f:
            sb = list(csv.DictReader(f))
        last = max(int(r["iter"]) for r in sb)
        sb = [r for r in sb if int(r["iter"]) == last]
        fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.4), gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]})

        # (a) stacked bars: gait share per commanded-speed bin, off-manifold in gray
        xb = np.arange(len(sb))
        bottom = np.zeros(len(sb))
        p_off = np.array([float(r["p_off"]) for r in sb])
        for g in gaits:
            v = np.nan_to_num(np.array([float(r["p_" + g]) for r in sb])) * (1 - p_off)
            axs[0].bar(xb, v, bottom=bottom, color=GAIT_COLORS[g], edgecolor="white", lw=0.6, label=LABELS[g])
            bottom += v
        axs[0].bar(xb, p_off, bottom=bottom, color=OFF_COLOR, edgecolor="white", lw=0.6, label="off-manifold")
        axs[0].set_xticks(xb)
        axs[0].set_xticklabels(["{:.1f}-{:.1f}\n(n={})".format(float(r["v_lo"]), float(r["v_hi"]), r["windows"]) for r in sb],
                               fontsize=8)
        axs[0].set_ylim(0, 1)
        axs[0].set_xlabel("commanded speed (m/s)")
        axs[0].set_ylabel("fraction of policy windows")
        axs[0].set_title("Gait per commanded speed (iter {})".format(last), loc="left", color=INK)
        axs[0].grid(axis="x", visible=False)
        axs[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.25), ncol=3, fontsize=8)

        # (b) commanded vs achieved speed, one dot per window, coloured by gait
        if "final_cmd" in d.files:
            ok = d["final_cmd_ok"].astype(bool)
            for i, g in enumerate(gaits):
                m = ok & d["final_on"] & (d["final_ff"] == i)
                if m.any():
                    axs[1].scatter(d["final_cmd"][m], d["final_v_along"][m], s=5, color=GAIT_COLORS[g], alpha=0.35, lw=0)
            m = ok & ~d["final_on"]
            if m.any():
                axs[1].scatter(d["final_cmd"][m], d["final_v_along"][m], s=5, color=OFF_COLOR, alpha=0.35, lw=0)
            lim = [0, float(d["speed_bins"][-1]) + 0.3]
            axs[1].plot(lim, lim, color=INK2, lw=0.8, ls="--")
            axs[1].text(lim[1], lim[1], "achieved = commanded", color=INK2, fontsize=8, ha="right", va="bottom")
            for v, name in [(1.1, "walk"), (1.9, "trot"), (3.2, "run")]:
                axs[1].axhline(v, color=INK2, lw=0.6, ls=":")
                axs[1].text(0.05, v, "ref. " + name, color=INK2, fontsize=7, va="bottom")
            axs[1].set_xlim(lim)
            axs[1].set_ylim(lim)
        axs[1].set_xlabel("commanded speed (m/s)")
        axs[1].set_ylabel("speed along target direction (m/s)")
        axs[1].set_title("Speed tracking (1-s windows)", loc="left", color=INK)

        # (c) I(command bin; gait) / H(gait) over training
        mi = [m for m in mets if "MI_norm" in m]
        if mi:
            # MI_norm is undefined (nan) when the policy uses a single gait (H(gait) = 0); that is the
            # "same gait at every speed" case, so plot it as 0 wherever the MI itself is defined
            xm, ym = align([m["samples"] for m in mi],
                           [np.where(np.isfinite(m["MI_cmd_gait"]), np.nan_to_num(m["MI_norm"]), np.nan) for m in mi])
            band(axs[2], xm / 1e6, ym, GAIT_COLORS["walk_pace"], "policy")
            axs[2].set_xlim(0, xm[-1] / 1e6)
        axs[2].set_ylim(0, 1)
        axs[2].set_xlabel("training samples (M)")
        axs[2].set_title("I(command; gait) / H(gait)", loc="left", color=INK)
        axs[2].text(0.02, 0.92, "0 = same gait at every speed\n1 = speed fully determines gait",
                    transform=axs[2].transAxes, color=INK2, fontsize=8, va="top")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "gait_by_speed.png"), bbox_inches="tight")
        plt.close(fig)

    print("Saved figures to", args.out)


if __name__ == "__main__":
    main()
