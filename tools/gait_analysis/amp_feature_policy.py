"""
Project trained policies into the morphology-agnostic AMP feature space fitted on the MoCap
(see amp_feature_clusters.py) and measure which motion modes they use. No footfall rule, no contact
threshold, no gait labels: modes come from clustering the AMP discriminator features of the MoCap.

Reference (MoCap only): AMP discriminator observation of 0.33 s windows -> k-means micro codes (K=32);
each 1 s window -> histogram of micro codes -> k-means macro modes (K=3). The macro modes are named after
the majority clip only for readability.

Policies: deterministic rollouts; each step records the policy's own 0.33 s AMP window (env._disc_obs_buf,
motion part only). A 1 s policy window takes the micro codes of its 21 short windows, exactly as the MoCap
windows, and goes to the nearest macro mode. A window is off-manifold when the median distance of its
short windows to the nearest micro centroid exceeds 1.5 x the MoCap 95th percentile.

Metrics per case: p(mode) over on-manifold windows, off-manifold fraction, normalized entropy over the 3
modes, JS divergence to the MoCap mode distribution (dataset weights), and for commanded runs
I(command; mode) / H(mode) (command = speed range of each gait for the baseline, c for the one-hot run).

Final checkpoints: location, steering baseline (all commands and per commanded speed range), run-only,
one-hot with c sampled as in training and with each c forced. Training dynamics: the same metrics at every
--every-th checkpoint of location, steering, run-only and one-hot (c sampled).

Outputs in --out_dir: amp_policy_modes.txt / .csv, amp_policy_training.csv and figures
amp_modes_baseline.png, amp_modes_cond.png, amp_modes_training.png, amp_metrics_training.png,
amp_policy_pca_baseline.png, amp_policy_pca_cond.png

Usage (from MimicKit/):
  python tools/gait_analysis/amp_feature_policy.py --out_dir output/figures/amp_features
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

import amp_feature_clusters as afc  # noqa: E402
import gait_collapse as gc  # noqa: E402
import envs.base_env as base_env  # noqa: E402
import envs.env_builder as env_builder  # noqa: E402
import learning.agent_builder as agent_builder  # noqa: E402
import learning.base_agent as base_agent  # noqa: E402
import util.mp_util as mp_util  # noqa: E402
import util.util as util  # noqa: E402

REF = gc.REF_GAITS
OFF_FACTOR = 1.5
S, L = 10, 30
SPEED_RANGES = {"walk_pace": (0.6, 1.4), "trot": (1.4, 2.2), "run": (2.4, 3.2)}


def build(run_dir, n, device):
    env = env_builder.build_env(os.path.join(run_dir, "env_config.yaml"), os.path.join(run_dir, "engine_config.yaml"),
                                n, device, visualize=False)
    agent = agent_builder.build_agent(os.path.join(run_dir, "agent_config.yaml"), env, device)
    return env, agent


def checkpoints(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "int_models", "model_*.pt")), key=gc.iter_of)


def fit_reference(env, k_micro, seed=0):
    dt = env._engine.get_timestep()
    ref = gc.build_reference(env, L)
    clip, start = ref["clip_ids"], ref["starts"]
    n = len(clip)
    clip_w = env._motion_lib.get_motion_weights()
    win_w = clip_w[clip] / torch.bincount(clip, minlength=len(clip_w)).float()[clip]
    win_w = win_w / win_w.mean()

    offs = torch.arange(S - 1, L, device=clip.device) * dt
    m_clip = clip[:, None].expand(-1, len(offs)).reshape(-1)
    m_end = (start[:, None] + offs[None, :]).reshape(-1)
    raw = afc.amp_windows(env, m_clip, m_end, S)
    mu, sd = raw.mean(0), raw.std(0).clamp_min(1e-3)
    x = (raw - mu) / sd
    code, cent = afc.kmeans(x, k_micro, w=win_w[:, None].expand(-1, len(offs)).reshape(-1), n_init=4, seed=seed)
    dmin = torch.cdist(x, cent).min(1).values
    off_thr = OFF_FACTOR * torch.quantile(dmin.view(n, -1).median(1).values, 0.95).item()

    bow = bow_of(code.view(n, -1), k_micro)
    lab, mcent = afc.kmeans(bow, 3, w=win_w, seed=seed)
    fg = ref["labels"].cpu().numpy()
    cont = afc.contingency(lab.cpu().numpy(), fg, 3, len(REF))
    names = [REF[int(np.argmax(cont[i] / cont.sum(0).clip(min=1)))] for i in range(3)]
    order = [names.index(g) for g in REF]                             # mode index for walk, trot, run
    remap = torch.tensor([REF.index(nm) for nm in names], device=clip.device)

    bmu = bow.mean(0, keepdim=True)
    _, _, vt = torch.linalg.svd(bow - bmu, full_matrices=False)
    basis = vt[:2].T
    q = np.array([win_w[lab == order[i]].sum().item() for i in range(3)])
    return {"mu": mu, "sd": sd, "cent": cent, "mcent": mcent, "remap": remap, "names": names,
            "off_thr": off_thr, "k": k_micro, "bmu": bmu, "basis": basis,
            "ref_pc": ((bow - bmu) @ basis).cpu().numpy(), "ref_gait": fg, "ref_dist": q / q.sum()}


def bow_of(codes, k):
    h = torch.zeros(codes.shape[0], k, device=codes.device)
    h.scatter_add_(1, codes, torch.ones_like(codes, dtype=torch.float))
    return torch.sqrt(h / h.sum(1, keepdim=True))


def rollout(env, agent, steps, warmup, settle):
    """1 s windows of the policy's 0.33 s AMP observations, [N, L-S+1, D], plus command metadata."""
    n, dev = env.get_num_envs(), env._device
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)
    obs, info = env.reset()
    D = env._disc_obs_buf.shape[-1]
    buf = torch.zeros([n, steps, D], device=dev)
    since = torch.zeros(n, device=dev, dtype=torch.long)
    since_hist = torch.zeros([n, steps], device=dev, dtype=torch.long)
    has_cmd = gc.has_speed_command(env)
    has_gait = hasattr(env, "get_gait_cmd")
    cmd = torch.zeros([n, steps], device=dev)
    gait = torch.zeros([n, steps], device=dev, dtype=torch.long)
    since_cmd = torch.zeros(n, device=dev, dtype=torch.long)
    since_cmd_hist = torch.zeros([n, steps], device=dev, dtype=torch.long)
    prev = torch.full([n], -1.0, device=dev)
    with torch.no_grad():
        for t in range(steps):
            if has_cmd:
                sp = env._tar_speed.clone()
                since_cmd[sp != prev] = 0
                prev = sp
            action, _ = agent._decide_action(obs, info)
            obs, _, done, info = env.step(action)
            buf[:, t] = env._disc_obs_buf
            since += 1
            since_hist[:, t] = since
            if has_cmd:
                since_cmd += 1
                cmd[:, t] = sp
                since_cmd_hist[:, t] = since_cmd
            if has_gait:
                gait[:, t] = env.get_gait_cmd()
            ids = (done != base_env.DoneFlags.NULL.value).nonzero().flatten()
            if len(ids) > 0:
                obs, info = env.reset(ids)
                since[ids] = 0
    ends = torch.arange(L - 1, steps, 3, device=dev)
    ok = since_hist[:, ends] >= max(L, warmup)
    e_i, t_i = ok.nonzero(as_tuple=True)
    t_end = ends[t_i]
    offs = torch.arange(-(L - S), 1, device=dev)
    out = {"win": buf[e_i[:, None], t_end[:, None] + offs[None, :]]}
    if has_cmd:
        out["cmd"] = cmd[e_i, t_end].cpu().numpy()
        out["cmd_ok"] = (since_cmd_hist[e_i, t_end] >= L + settle).cpu().numpy()
    if has_gait:
        out["gait"] = gait[e_i, t_end].cpu().numpy()
    return out


def assign(R, win):
    n, m, D = win.shape
    modes, offs, pcs = [], [], []
    for i in range(0, n, 2048):
        w = win[i:i + 2048]
        x = (w.reshape(-1, D) - R["mu"]) / R["sd"]
        dmin, code = torch.cdist(x, R["cent"]).min(1)
        offs.append(dmin.view(len(w), m).median(1).values > R["off_thr"])
        bow = bow_of(code.view(len(w), m), R["k"])
        modes.append(R["remap"][torch.cdist(bow, R["mcent"]).argmin(1)])
        pcs.append((bow - R["bmu"]) @ R["basis"])
    return torch.cat(modes).cpu().numpy(), torch.cat(offs).cpu().numpy(), torch.cat(pcs).cpu().numpy()


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def summarize(name, R, mode, off, mask=None, cond=None):
    """mode indexes REF (walk, trot, run). cond: per-window command class for I(command; mode)."""
    mask = np.ones_like(off) if mask is None else mask
    on = mask & ~off
    p = np.bincount(mode[on], minlength=3) / max(on.sum(), 1)
    q = R["ref_dist"]
    mm = 0.5 * (p + q)
    kl = lambda a, b: float(sum(a[i] * np.log2(a[i] / b[i]) for i in range(3) if a[i] > 0))
    row = {"case": name, "n": int(mask.sum()), "off": float(1 - on.sum() / max(mask.sum(), 1)),
           **{"p_" + g: float(p[i]) for i, g in enumerate(REF)},
           "H_norm": entropy(p) / np.log(3) if on.any() else float("nan"),
           "JS": 0.5 * kl(p, mm) + 0.5 * kl(q, mm) if on.any() else float("nan"), "MI_norm": float("nan")}
    if cond is not None:
        c, m = cond[on], mode[on]
        valid = c >= 0
        if valid.sum() > 0:
            joint = afc.contingency(c[valid], m[valid], c.max() + 1, 3) / valid.sum()
            pc, pm = joint.sum(1), joint.sum(0)
            nz = joint > 0
            mi = (joint[nz] * np.log(joint[nz] / (pc[:, None] * pm[None, :])[nz])).sum()
            row["MI_norm"] = float(mi / entropy(pm)) if entropy(pm) > 0 else 0.0
    return row


def speed_class(cmd, ok):
    c = np.full(len(cmd), -1)
    for i, g in enumerate(REF):
        lo, hi = SPEED_RANGES[g]
        c[ok & (cmd >= lo) & (cmd <= hi)] = i
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default="output/go2_loc_amp_allgaits_s42")
    ap.add_argument("--baseline", default="output/go2_steer_amp_allgaits_2048env_s42")
    ap.add_argument("--cond", default="output/go2_steer_gaitcond_2048env_s42")
    ap.add_argument("--run_only", default="output/go2_amp_run_only_2048env_s42")
    ap.add_argument("--out_dir", default="output/figures/amp_features")
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--rollout_sec", type=float, default=20.0)
    ap.add_argument("--dyn_envs", type=int, default=64)
    ap.add_argument("--dyn_sec", type=float, default=10.0)
    ap.add_argument("--every", type=int, default=1, help="training dynamics: every n-th checkpoint")
    ap.add_argument("--warmup_sec", type=float, default=2.0)
    ap.add_argument("--settle_sec", type=float, default=1.0)
    ap.add_argument("--k_micro", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    mp_util.init(0, 1, args.device, None)

    rows, pcs, dyn = [], {}, {}
    env, agent = build(args.baseline, args.num_envs, args.device)
    dt = env._engine.get_timestep()
    steps, warm, settle = int(args.rollout_sec / dt), int(args.warmup_sec / dt), int(args.settle_sec / dt)
    R = fit_reference(env, args.k_micro)
    print("Macro modes named by majority clip:", R["names"], "MoCap distribution:", R["ref_dist"])

    def run_case(env, agent, ckpt, n_steps, forced=None):
        agent.load(ckpt)
        if hasattr(env, "set_forced_gait"):
            env.set_forced_gait(forced)
        util.set_rand_seed(0)
        return rollout(env, agent, n_steps, warm, settle)

    def dynamics(key, env, agent, run_dir):
        dsteps = int(args.dyn_sec / dt)
        samples = gc.samples_by_iter(run_dir)
        out = []
        for ck in checkpoints(run_dir)[::args.every]:
            it = gc.iter_of(ck)
            r = run_case(env, agent, ck, dsteps)
            mode, off, _ = assign(R, r["win"])
            row = summarize(key, R, mode, off, cond=r.get("gait"))
            row["iter"], row["samples"] = it, samples.get(it, it * 131072)
            out.append(row)
            print("  {} iter {:5d}: walk {:.0%} trot {:.0%} run {:.0%} off {:.0%}".format(
                key, it, row["p_walk_pace"], row["p_trot"], row["p_run"], row["off"]))
        dyn[key] = out

    # ---------------- steering baseline
    r = run_case(env, agent, os.path.join(args.baseline, "model.pt"), steps)
    mode, off, pc = assign(R, r["win"])
    sc = speed_class(r["cmd"], r["cmd_ok"])
    rows.append(summarize("steering, all commands", R, mode, off, cond=sc))
    pcs["steering"] = (pc, off)
    for i, g in enumerate(REF):
        lo, hi = SPEED_RANGES[g]
        rows.append(summarize("steering, cmd {}-{} m/s".format(lo, hi), R, mode, off, mask=sc == i))
    env._agent_for_dyn = None
    sub_env, sub_agent = build(args.baseline, args.dyn_envs, args.device)
    dynamics("steering", sub_env, sub_agent, args.baseline)
    del env, agent, sub_env, sub_agent
    torch.cuda.empty_cache()

    # ---------------- location
    env, agent = build(args.location, args.num_envs, args.device)
    r = run_case(env, agent, os.path.join(args.location, "model.pt"), steps)
    mode, off, pc = assign(R, r["win"])
    rows.append(summarize("location (1 m/s)", R, mode, off))
    pcs["location"] = (pc, off)
    del env, agent
    env, agent = build(args.location, args.dyn_envs, args.device)
    dynamics("location", env, agent, args.location)
    del env, agent
    torch.cuda.empty_cache()

    # ---------------- run only
    env, agent = build(args.run_only, args.num_envs, args.device)
    r = run_case(env, agent, os.path.join(args.run_only, "model.pt"), steps)
    mode, off, pc = assign(R, r["win"])
    rows.append(summarize("run clip only", R, mode, off))
    pcs["run-only"] = (pc, off)
    del env, agent
    env, agent = build(args.run_only, args.dyn_envs, args.device)
    dynamics("run-only", env, agent, args.run_only)
    del env, agent
    torch.cuda.empty_cache()

    # ---------------- one-hot conditioned
    env, agent = build(args.cond, args.num_envs, args.device)
    r = run_case(env, agent, os.path.join(args.cond, "model.pt"), steps, forced=None)
    mode, off, pc = assign(R, r["win"])
    rows.append(summarize("one-hot, c sampled", R, mode, off, cond=r["gait"]))
    for k, g in enumerate(env.get_gait_classes()):
        r = run_case(env, agent, os.path.join(args.cond, "model.pt"), steps, forced=k)
        mode, off, pc = assign(R, r["win"])
        rows.append(summarize("one-hot, c = " + g, R, mode, off))
        pcs["one-hot, c = " + g] = (pc, off)
    del env, agent
    env, agent = build(args.cond, args.dyn_envs, args.device)
    dynamics("one-hot", env, agent, args.cond)
    del env, agent
    torch.cuda.empty_cache()

    # ---------------- report
    lines = ["Policies projected into the AMP feature space fitted on the MoCap (micro codes K={}, macro modes K=3)".format(args.k_micro),
             "MoCap mode distribution (dataset weights): " + ", ".join("{}={:.2f}".format(g, v) for g, v in zip(REF, R["ref_dist"])),
             "p(mode) over on-manifold 1 s windows; off = far from every micro code; MI_norm = I(command; mode) / H(mode)", "",
             "{:26s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>7s} {:>6s} {:>8s}".format(
                 "case", "n", "walk", "trot", "run", "off", "H_norm", "JS", "MI_norm")]
    for w in rows:
        lines.append("{:26s} {:6d} {:6.0%} {:6.0%} {:6.0%} {:6.0%} {:7.2f} {:6.3f} {:8.2f}".format(
            w["case"], w["n"], w["p_walk_pace"], w["p_trot"], w["p_run"], w["off"], w["H_norm"], w["JS"], w["MI_norm"]))
    lines.append("")
    lines.append("Training dynamics (every {} checkpoint(s), {} envs x {:.0f} s):".format(args.every, args.dyn_envs, args.dyn_sec))
    for key, out in dyn.items():
        lines.append("  " + key + ": " + "  ".join("{:.0f}M:{:.0%}/{:.0%}/{:.0%}".format(
            o["samples"] / 1e6, o["p_walk_pace"], o["p_trot"], o["p_run"]) for o in out))
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out_dir, "amp_policy_modes.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(args.out_dir, "amp_policy_modes.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(args.out_dir, "amp_policy_training.csv"), "w", newline="") as f:
        allrows = [o for out in dyn.values() for o in out]
        wr = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        wr.writeheader()
        wr.writerows(allrows)

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_gait_collapse as pg

    byname = {w["case"]: w for w in rows}
    mocap = {"p_" + g: float(v) for g, v in zip(REF, R["ref_dist"])} | {"off": 0.0}

    def bars(items, path, title):
        fig, ax = plt.subplots(figsize=(1.35 * len(items) + 2.5, 3.8))
        for i, (lbl, w) in enumerate(items):
            bottom = 0.0
            for g in REF:
                v = w["p_" + g] * (1 - w["off"])
                ax.bar(i, v, 0.7, bottom=bottom, color=pg.GAIT_COLORS[g], edgecolor="white", lw=0.6,
                       label="mode " + pg.LABELS[g] if i == 0 else None)
                bottom += v
            ax.bar(i, w["off"], 0.7, bottom=bottom, color=pg.OFF_COLOR, edgecolor="white", lw=0.6,
                   label="off-manifold" if i == 0 else None)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([b[0] for b in items], fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("fraction of 1 s windows")
        ax.set_title(title, loc="left", color=pg.INK, fontsize=11)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, path), bbox_inches="tight", dpi=200)
        plt.close(fig)

    bars([("MoCap\n(dataset)", mocap), ("location\n1 m/s", byname["location (1 m/s)"]),
          ("steering\ncmd 0.6-1.4", byname["steering, cmd 0.6-1.4 m/s"]),
          ("steering\ncmd 1.4-2.2", byname["steering, cmd 1.4-2.2 m/s"]),
          ("steering\ncmd 2.4-3.2", byname["steering, cmd 2.4-3.2 m/s"]),
          ("only the\nrun clip", byname["run clip only"])],
         "amp_modes_baseline.png", "Modes used by vanilla AMP policies (AMP feature space)")
    bars([("MoCap\n(dataset)", mocap), ("one-hot\nc sampled", byname["one-hot, c sampled"]),
          ("one-hot\nc = walk", byname["one-hot, c = walk_pace"]), ("one-hot\nc = trot", byname["one-hot, c = trot"]),
          ("one-hot\nc = run", byname["one-hot, c = run"])],
         "amp_modes_cond.png", "Modes used by the conditioned policy (AMP feature space)")

    titles = {"location": "location (1 m/s)", "steering": "steering (0.6-3.2 m/s)", "run-only": "only the run clip",
              "one-hot": "steering + one-hot condition"}
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.6), sharey=True)
    for ax, key in zip(axs, ["location", "steering", "run-only", "one-hot"]):
        out = dyn[key]
        x = np.array([o["samples"] for o in out]) / 1e6
        ys = [np.array([o["p_" + g] * (1 - o["off"]) for o in out]) for g in REF] + [np.array([o["off"] for o in out])]
        ax.stackplot(x, *ys, colors=[pg.GAIT_COLORS[g] for g in REF] + [pg.OFF_COLOR],
                     labels=["mode " + pg.LABELS[g] for g in REF] + ["off-manifold"], edgecolor="white", lw=0.4)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, 1)
        ax.set_title(titles[key], loc="left", color=pg.INK, fontsize=11)
        ax.set_xlabel("training samples (M)")
    axs[0].set_ylabel("fraction of 1 s windows")
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=10)
    fig.savefig(os.path.join(args.out_dir, "amp_modes_training.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
    cols = {"location": "#8a5cd1", "steering": "#2a78d6", "run-only": "#1baf7a", "one-hot": "#eb6834"}
    for key, out in dyn.items():
        x = np.array([o["samples"] for o in out]) / 1e6
        axs[0].plot(x, [o["H_norm"] for o in out], color=cols[key], lw=2, label=titles[key])
        axs[1].plot(x, [o["JS"] for o in out], color=cols[key], lw=2, label=titles[key])
        if key == "one-hot":
            # only checkpoints where most windows are on the manifold (early ones are mostly falling)
            mi = [o["MI_norm"] if o["off"] < 0.5 else float("nan") for o in out]
            axs[2].plot(x, mi, color=cols[key], lw=2, marker="o", ms=3, label="one-hot: I(c; mode) / H(mode)")
    q = R["ref_dist"]
    axs[0].axhline(entropy(q) / np.log(3), color=pg.INK2, ls="--", lw=1)
    axs[0].text(0.98, entropy(q) / np.log(3) + 0.02, "MoCap", color=pg.INK2, fontsize=9, ha="right",
                transform=axs[0].get_yaxis_transform())
    axs[0].set_title("Entropy over the modes", loc="left", color=pg.INK, fontsize=11)
    axs[1].set_title("JS divergence to the MoCap modes", loc="left", color=pg.INK, fontsize=11)
    axs[2].set_title("Mode explained by the command", loc="left", color=pg.INK, fontsize=11)
    st = byname["steering, all commands"]["MI_norm"]
    axs[2].axhline(st, color=cols["steering"], ls="--", lw=1.5, label="steering final: I(speed range; mode) / H")
    for ax in axs:
        ax.set_xlabel("training samples (M)")
        ax.set_ylim(0, 1.05)
    axs[0].legend(frameon=False, fontsize=8)
    axs[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "amp_metrics_training.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    def pca_fig(keys, ttls, path):
        fig, axs = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.2), sharex=True, sharey=True)
        rp = R["ref_pc"]
        for ax, key, ttl in zip(axs, keys, ttls):
            for gi, g in enumerate(REF):
                m = R["ref_gait"] == gi
                ax.scatter(rp[m, 0], rp[m, 1], s=12, color=pg.GAIT_COLORS[g], alpha=0.35, lw=0,
                           label="MoCap " + pg.LABELS[g] if key == keys[0] else None)
            pc, off = pcs[key]
            ax.scatter(pc[off, 0], pc[off, 1], s=4, color=pg.OFF_COLOR, alpha=0.4, lw=0,
                       label="policy, off-manifold" if key == keys[0] else None)
            ax.scatter(pc[~off, 0], pc[~off, 1], s=4, color=pg.INK, alpha=0.3, lw=0,
                       label="policy" if key == keys[0] else None)
            ax.set_title(ttl, loc="left", color=pg.INK, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel("PC1")
        axs[0].set_ylabel("PC2")
        h, l = axs[0].get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False, fontsize=10, markerscale=3)
        fig.savefig(os.path.join(args.out_dir, path), bbox_inches="tight", dpi=200)
        plt.close(fig)

    pca_fig(["location", "steering", "run-only"], ["location (1 m/s)", "steering (0.6-3.2 m/s)", "only the run clip"],
            "amp_policy_pca_baseline.png")
    pca_fig(["one-hot, c = walk_pace", "one-hot, c = trot", "one-hot, c = run"],
            ["one-hot, c = walk", "one-hot, c = trot", "one-hot, c = run"], "amp_policy_pca_cond.png")
    print("Saved figures to", args.out_dir)


if __name__ == "__main__":
    main()
