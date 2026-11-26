"""
P(gait | commanded gait) for a gait-conditioned run (gait_cond_steering env), compared with the
unconditioned steering baseline at the same commanded speeds.

Conditioned run: for each gait class c, every env is reset with c forced (episodes start in a clip
of class c, the target speed is drawn from the speed range of c) and the policy is rolled out with
deterministic actions. Each 1 s window is labelled by the footfall classifier of gait_collapse.py;
windows far from all reference motion (kNN distance > 1.5 x reference p95) count as off-manifold.

Baseline (--baseline_dir): one rollout of the unconditioned steering policy; windows are grouped by
commanded speed into the speed range of each class (command constant for the window + settle time),
so both columns of a class face the same task.

Outputs in <run_dir>/cond_eval/: cond_gait.csv, cond_gait.txt, cond_gait.png

Usage (from MimicKit/):
  python tools/gait_analysis/cond_gait_eval.py --run_dir output/go2_steer_gaitcond_2048env_s42 \
      --baseline_dir output/go2_steer_amp_allgaits_2048env_s42
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "mimickit"))

import gait_collapse as gc  # noqa: E402
import envs.env_builder as env_builder  # noqa: E402
import learning.agent_builder as agent_builder  # noqa: E402
import util.mp_util as mp_util  # noqa: E402
import util.util as util  # noqa: E402


def load(run_dir, num_envs, device, ckpt=None):
    env = env_builder.build_env(os.path.join(run_dir, "env_config.yaml"), os.path.join(run_dir, "engine_config.yaml"),
                                num_envs, device, visualize=False)
    agent = agent_builder.build_agent(os.path.join(run_dir, "agent_config.yaml"), env, device)
    if ckpt is None:
        ckpts = sorted(glob.glob(os.path.join(run_dir, "int_models", "model_*.pt")), key=gc.iter_of)
        ckpt = ckpts[-1] if ckpts else os.path.join(run_dir, "model.pt")
    agent.load(ckpt)
    return env, agent, ckpt


def reference_tools(env, win_steps, nd, win_sec, sub, k):
    ref = gc.build_reference(env, win_steps)
    clf = gc.KNNGait(ref["feats"], ref["labels"], sub, k)
    clip, start = ref["clip_ids"], ref["starts"]
    excl = lambda a, b: (clip[a:b, None] == clip[None, :]) & ((start[a:b, None] - start[None, :]).abs() < win_sec)
    _, ref_nn = clf.query(ref["feats"], excl)
    off_thr = gc.OFF_MANIFOLD_FACTOR * torch.quantile(ref_nn[torch.isfinite(ref_nn)], 0.95).item()
    return clf, off_thr


def classify(r, nd, clf, off_thr):
    st = gc.footfall_stats(r["windows"], nd, gc.resting_foot_height(r["windows"], nd))
    ff = gc.footfall_classify(st).cpu().numpy()
    _, nn = clf.query(r["windows"])
    on = nn.cpu().numpy() <= off_thr
    return ff, on, st


def summarize(ff, on, mask, speed):
    m = mask & on
    n_all = int(mask.sum())
    p = np.bincount(ff[m], minlength=len(gc.GAITS)) / max(m.sum(), 1)
    return {"n": n_all, "p_off": float(1 - m.sum() / max(n_all, 1)), **{"p_" + g: float(v) for g, v in zip(gc.GAITS, p)},
            "speed": float(speed[mask].mean()) if n_all else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="gait-conditioned run")
    ap.add_argument("--baseline_dir", default=None, help="unconditioned steering run, same task")
    ap.add_argument("--model", default=None, help="checkpoint of the conditioned run (default: last int_model)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--rollout_sec", type=float, default=20.0)
    ap.add_argument("--warmup_sec", type=float, default=2.0)
    ap.add_argument("--win_sec", type=float, default=1.0)
    ap.add_argument("--stride_steps", type=int, default=3)
    ap.add_argument("--settle_sec", type=float, default=1.0)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--sub", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, "cond_eval")
    os.makedirs(out_dir, exist_ok=True)
    mp_util.init(0, 1, args.device, None)

    env, agent, ckpt = load(args.run_dir, args.num_envs, args.device, args.model)
    classes = env.get_gait_classes()
    lo, hi = env._gait_speed_lo.cpu().numpy(), env._gait_speed_hi.cpu().numpy()
    dt = env._engine.get_timestep()
    nd = env._engine.get_dof_pos(env._get_char_id()).shape[-1]
    win, steps = int(round(args.win_sec / dt)), int(round(args.rollout_sec / dt))
    warm, settle = int(round(args.warmup_sec / dt)), int(round(args.settle_sec / dt))
    clf, off_thr = reference_tools(env, win, nd, args.win_sec, args.sub, args.knn)

    rows = []
    for k, c in enumerate(classes):
        util.set_rand_seed(args.seed)
        env.set_forced_gait(k)
        r = gc.rollout(env, agent, steps, win, warm, args.stride_steps, settle)
        ff, on, st = classify(r, nd, clf, off_thr)
        speed = st["speed"].cpu().numpy()
        row = {"model": "conditioned", "cmd_gait": c, "speed_range": "{:.1f}-{:.1f}".format(lo[k], hi[k]),
               **summarize(ff, on, np.ones_like(on), speed),
               "cmd_speed": float(r["cmd"].mean()), "v_along": float(r["v_along"].mean()),
               "falls_per_ep": r["falls"] / r["episodes"]}
        rows.append(row)
    env.set_forced_gait(None)
    del env, agent
    torch.cuda.empty_cache()

    if args.baseline_dir:
        benv, bagent, bckpt = load(args.baseline_dir, args.num_envs, args.device)
        util.set_rand_seed(args.seed)
        r = gc.rollout(benv, bagent, steps, win, warm, args.stride_steps, settle)
        ff, on, st = classify(r, nd, clf, off_thr)
        speed = st["speed"].cpu().numpy()
        cmd, ok, va = r["cmd"].cpu().numpy(), r["cmd_ok"].cpu().numpy(), r["v_along"].cpu().numpy()
        for k, c in enumerate(classes):
            mask = ok & (cmd >= lo[k]) & (cmd <= hi[k])
            rows.append({"model": "baseline", "cmd_gait": c, "speed_range": "{:.1f}-{:.1f}".format(lo[k], hi[k]),
                         **summarize(ff, on, mask, speed),
                         "cmd_speed": float(cmd[mask].mean()), "v_along": float(va[mask].mean()),
                         "falls_per_ep": r["falls"] / r["episodes"]})

    keys = list(rows[0].keys())
    with open(os.path.join(out_dir, "cond_gait.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    lines = ["Conditioned run: {}  ({})".format(args.run_dir, os.path.basename(ckpt))]
    if args.baseline_dir:
        lines.append("Baseline: {}  ({}), windows grouped by commanded speed".format(args.baseline_dir, os.path.basename(bckpt)))
    lines.append("p(gait) over on-manifold windows; speed = mean root speed of the windows (m/s)")
    lines.append("")
    hdr = "{:12s} {:10s} {:8s} {:>6s} " + " ".join("{:>9s}" for _ in gc.GAITS) + " {:>6s} {:>6s} {:>7s} {:>7s} {:>6s}"
    lines.append(hdr.format("model", "cmd_gait", "v_range", "n", *gc.GAITS, "off", "speed", "v_cmd", "v_along", "falls"))
    for r in rows:
        lines.append(("{:12s} {:10s} {:8s} {:6d} " + " ".join("{:9.0%}" for _ in gc.GAITS) +
                      " {:6.0%} {:6.2f} {:7.2f} {:7.2f} {:6.2f}").format(
            r["model"], r["cmd_gait"], r["speed_range"], r["n"], *[r["p_" + g] for g in gc.GAITS],
            r["p_off"], r["speed"], r["cmd_speed"], r["v_along"], r["falls_per_ep"]))
    cond = [r for r in rows if r["model"] == "conditioned"]
    acc = np.mean([r["p_" + r["cmd_gait"]] for r in cond])
    lines.append("")
    lines.append("Conditioned: mean P(gait = commanded) = {:.0%}".format(acc))
    if args.baseline_dir:
        base = [r for r in rows if r["model"] == "baseline"]
        lines.append("Baseline, same speed ranges: mean P(gait = class of the range) = {:.0%}".format(
            np.mean([r["p_" + r["cmd_gait"]] for r in base])))
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, "cond_gait.txt"), "w") as f:
        f.write(report + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_gait_collapse as pg

    models = ["baseline", "conditioned"] if args.baseline_dir else ["conditioned"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    width, xt, xl = 0.38, [], []
    for k, c in enumerate(classes):
        for j, m in enumerate(models):
            r = next(x for x in rows if x["model"] == m and x["cmd_gait"] == c)
            x = k + (j - 0.5 * (len(models) - 1)) * (width + 0.04)
            bottom = 0.0
            for g in gc.GAITS:
                v = r["p_" + g] * (1 - r["p_off"])
                ax.bar(x, v, width, bottom=bottom, color=pg.GAIT_COLORS[g], edgecolor="white", lw=0.6,
                       label=pg.LABELS.get(g, g) if (k == 0 and j == 0) else None)
                bottom += v
            ax.bar(x, r["p_off"], width, bottom=bottom, color=pg.OFF_COLOR, edgecolor="white", lw=0.6,
                   label="off-manifold" if (k == 0 and j == 0) else None)
            ax.text(x, 1.02, "base" if m == "baseline" else "cond", ha="center", fontsize=8, color=pg.INK2)
        xt.append(k)
        xl.append("{}\n({} m/s)".format(c, "{:.1f}-{:.1f}".format(lo[k], hi[k])))
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("fraction of windows")
    ax.set_title("Gait produced for each commanded gait / speed range", loc="left", color=pg.INK, fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cond_gait.png"), dpi=200, bbox_inches="tight")
    print("Saved", os.path.join(out_dir, "cond_gait.png"))


if __name__ == "__main__":
    main()
