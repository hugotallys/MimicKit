"""
Project trained policies onto the motion VQ-VAE codes fitted on the MoCap (train.py) and measure which
motion modes they use. Same cases and metrics as tools/gait_analysis/amp_feature_policy.py, with the
VQ-VAE tokens in place of the k-means micro codes, so the two representations can be compared.

Rollouts are deterministic; every step records the newest simulated state of each robot (the AMP
discriminator history), and W-step windows are rebuilt with the AMP feature function (features.py).
A window: tokens -> Hellinger histogram -> nearest macro mode of the MoCap; off-manifold when its
reconstruction MSE exceeds 1.5 x the MoCap 95th percentile (the VQ-VAE cannot rebuild unseen motion). Windows overlapping a reset are dropped.

Outputs in --out_dir: vqvae_policy_modes.txt / .csv, vqvae_modes_policies.png, vqvae_policy_pca.png

Usage (from MimicKit/):
  python tools/motion_vqvae/eval_policies.py --ckpt output/motion_vqvae/vqvae_go2_allgaits.pt
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "gait_analysis"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "mimickit"))

import amp_feature_policy as afp  # noqa: E402
import features as feat  # noqa: E402
import gait_collapse as gc  # noqa: E402
import envs.base_env as base_env  # noqa: E402
import learning.base_agent as base_agent  # noqa: E402
import util.mp_util as mp_util  # noqa: E402
import util.util as util  # noqa: E402
from model import MotionVQVAE  # noqa: E402
from train import hellinger_hist  # noqa: E402

REF = gc.REF_GAITS


def load(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = MotionVQVAE(**ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ck["model"] = model
    ck["remap"] = torch.tensor([REF.index(n) for n in ck["macro_names"]], device=device)
    return ck


def rollout(env, agent, steps, W, warmup, settle, check_step=60):
    n, dev = env.get_num_envs(), env._device
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)
    obs, info = env.reset()
    rec = feat.StateRecorder(env, steps)
    since = torch.zeros(n, device=dev, dtype=torch.long)
    since_hist = torch.zeros([n, steps], device=dev, dtype=torch.long)
    has_cmd, has_gait = gc.has_speed_command(env), hasattr(env, "get_gait_cmd")
    cmd = torch.zeros([n, steps], device=dev)
    gait = torch.zeros([n, steps], device=dev, dtype=torch.long)
    since_cmd = torch.zeros(n, device=dev, dtype=torch.long)
    since_cmd_hist = torch.zeros([n, steps], device=dev, dtype=torch.long)
    prev = torch.full([n], -1.0, device=dev)
    check = None
    with torch.no_grad():
        for t in range(steps):
            if has_cmd:
                sp = env._tar_speed.clone()
                since_cmd[sp != prev] = 0
                prev = sp
            action, _ = agent._decide_action(obs, info)
            obs, _, done, info = env.step(action)
            rec.record(t)
            since += 1
            since_hist[:, t] = since
            if t == check_step:
                check = env._disc_obs_buf.clone()
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

    # the rebuilt 10-step window must equal the env's own discriminator observation
    S = env._num_disc_obs_steps
    ok_chk = (since_hist[:, check_step] >= S).nonzero().flatten()
    if len(ok_chk) > 0:
        mine = rec.windows(ok_chk, torch.full_like(ok_chk, check_step), S).reshape(len(ok_chk), -1)
        err = (mine - check[ok_chk]).abs().max().item()
        assert err < 1e-4, "rebuilt AMP window differs from env._disc_obs_buf (max abs {:.2e})".format(err)

    ends = torch.arange(W - 1, steps, 3, device=dev)
    ok = since_hist[:, ends] >= max(W, warmup)
    e_i, t_i = ok.nonzero(as_tuple=True)
    t_end = ends[t_i]
    out = {"win": rec.windows(e_i, t_end, W)}
    if has_cmd:
        out["cmd"] = cmd[e_i, t_end].cpu().numpy()
        out["cmd_ok"] = (since_cmd_hist[e_i, t_end] >= W + settle).cpu().numpy()
    if has_gait:
        out["gait"] = gait[e_i, t_end].cpu().numpy()
    return out


def assign(ck, win):
    """Mode of every window (no off-manifold filter), and the two off-manifold flags."""
    model = ck["model"]
    modes, off_lat, off_rec, pcs = [], [], [], []
    for i in range(0, len(win), 4096):
        x = ((win[i:i + 4096] - ck["mu"]) / ck["sd"]).permute(0, 2, 1).contiguous()
        codes, _, qd = model.encode(x)
        with torch.no_grad():
            recon = model(x)[0]
        off_lat.append(qd.median(1).values > ck["off_thr"])
        off_rec.append((recon - x).pow(2).mean((1, 2)) > ck["off_thr_rec"])
        h = hellinger_hist(codes, model.config["num_embeddings"])
        modes.append(ck["remap"][torch.cdist(h, ck["macro_centroids"]).argmin(1)])
        pcs.append((h - ck["hist_mu"]) @ ck["hist_basis"])
    cat = lambda v: torch.cat(v).cpu().numpy()
    return cat(modes), (cat(off_lat), cat(off_rec)), cat(pcs)


def summ(name, R, mode, offs, mask=None, cond=None):
    m = np.ones(len(mode), dtype=bool) if mask is None else mask
    row = afp.summarize(name, R, mode, np.zeros(len(mode), dtype=bool), mask=m, cond=cond)
    row.pop("off")
    row["off_latent"] = float(offs[0][m].mean()) if m.any() else float("nan")
    row["off_recon"] = float(offs[1][m].mean()) if m.any() else float("nan")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="output/motion_vqvae/vqvae_go2_allgaits.pt")
    ap.add_argument("--location", default="output/go2_loc_amp_allgaits_s42")
    ap.add_argument("--baseline", default="output/go2_steer_amp_allgaits_2048env_s42")
    ap.add_argument("--cond", default="output/go2_steer_gaitcond_2048env_s42")
    ap.add_argument("--run_only", default="output/go2_amp_run_only_2048env_s42")
    ap.add_argument("--kmeans_csv", default="output/figures/amp_features/amp_policy_modes.csv",
                    help="results of amp_feature_policy.py, printed side by side")
    ap.add_argument("--out_dir", default="output/figures/motion_vqvae")
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--rollout_sec", type=float, default=20.0)
    ap.add_argument("--warmup_sec", type=float, default=2.0)
    ap.add_argument("--settle_sec", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    mp_util.init(0, 1, args.device, None)
    ck = load(args.ckpt, args.device)
    W = ck["window"]
    R = {"ref_dist": np.asarray(ck["ref_dist"])}
    rows, pcs = [], {}

    def case(run_dir, forced_list=(None,)):
        env, agent = afp.build(run_dir, args.num_envs, args.device)
        dt = env._engine.get_timestep()
        steps, warm, settle = int(args.rollout_sec / dt), int(args.warmup_sec / dt), int(args.settle_sec / dt)
        out = []
        for forced in forced_list:
            agent.load(os.path.join(run_dir, "model.pt"))
            if hasattr(env, "set_forced_gait"):
                env.set_forced_gait(forced)
            util.set_rand_seed(0)
            r = rollout(env, agent, steps, W, warm, settle)
            out.append((forced, r, *assign(ck, r["win"])))
        names = env.get_gait_classes() if hasattr(env, "get_gait_classes") else None
        del env, agent
        torch.cuda.empty_cache()
        return out, names

    (_, r, mode, off, pc), = case(args.baseline)[0]
    sc = afp.speed_class(r["cmd"], r["cmd_ok"])
    rows.append(summ("steering, all commands", R, mode, off, cond=sc))
    for i, g in enumerate(REF):
        lo, hi = afp.SPEED_RANGES[g]
        rows.append(summ("steering, cmd {}-{} m/s".format(lo, hi), R, mode, off, mask=sc == i))
    pcs["steering"] = (pc, off)
    (_, r, mode, off, pc), = case(args.location)[0]
    rows.append(summ("location (1 m/s)", R, mode, off))
    pcs["location"] = (pc, off)
    (_, r, mode, off, pc), = case(args.run_only)[0]
    rows.append(summ("run clip only", R, mode, off))
    pcs["run-only"] = (pc, off)
    out, gnames = case(args.cond, forced_list=(None, 0, 1, 2))
    for forced, r, mode, off, pc in out:
        if forced is None:
            rows.append(summ("one-hot, c sampled", R, mode, off, cond=r["gait"]))
        else:
            rows.append(summ("one-hot, c = " + gnames[forced], R, mode, off))
            pcs["one-hot, c = " + gnames[forced]] = (pc, off)

    km = {}
    if os.path.exists(args.kmeans_csv):
        with open(args.kmeans_csv) as f:
            km = {w["case"]: w for w in csv.DictReader(f)}
    lines = ["Policies projected onto the motion VQ-VAE codes (W = {} frames, K = {}, {} tokens per window)".format(
                 W, ck["config"]["num_embeddings"], W // 2 ** ck["config"]["n_down"]),
             "MoCap mode distribution (dataset weights): " + ", ".join("{}={:.2f}".format(g, v) for g, v in zip(REF, R["ref_dist"])),
             "p(mode) over ALL windows (nearest macro mode of the token histogram); off-manifold reported, not filtered:",
             "  off_lat = median token-to-code distance > 1.5 x MoCap p95; off_rec = reconstruction MSE > 1.5 x MoCap p95",
             "MI_norm = I(command; mode) / H(mode); [k-means] = same case with the k-means micro codes (amp_feature_policy.py)", "",
             "{:26s} {:>6s} {:>6s} {:>6s} {:>8s} {:>8s} {:>7s} {:>6s} {:>8s}   {:>22s}".format(
                 "case", "walk", "trot", "run", "off_lat", "off_rec", "H_norm", "JS", "MI_norm", "[k-means] w/t/r off MI")]
    for w in rows:
        k = km.get(w["case"])
        ks = "{:4.0%}/{:4.0%}/{:4.0%} {:4.0%} {:5.2f}".format(float(k["p_walk_pace"]), float(k["p_trot"]), float(k["p_run"]),
                                                            float(k["off"]), float(k["MI_norm"])) if k else ""
        lines.append("{:26s} {:6.0%} {:6.0%} {:6.0%} {:8.0%} {:8.0%} {:7.2f} {:6.3f} {:8.2f}   {}".format(
            w["case"], w["p_walk_pace"], w["p_trot"], w["p_run"], w["off_latent"], w["off_recon"], w["H_norm"], w["JS"],
            w["MI_norm"], ks))
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out_dir, "vqvae_policy_modes.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(args.out_dir, "vqvae_policy_modes.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_gait_collapse as pg

    byname = {w["case"]: w for w in rows}
    mocap = {"p_" + g: float(v) for g, v in zip(REF, R["ref_dist"])} | {"off_recon": 0.0}
    items = [("MoCap\n(dataset)", mocap), ("location\n1 m/s", byname["location (1 m/s)"]),
             ("steering\nall cmds", byname["steering, all commands"]), ("only the\nrun clip", byname["run clip only"]),
             ("one-hot\nc sampled", byname["one-hot, c sampled"])] + [
            ("one-hot\nc = " + pg.LABELS[g].split(" ")[0], byname["one-hot, c = " + g]) for g in gnames]
    fig, ax = plt.subplots(figsize=(1.3 * len(items) + 2.5, 3.8))
    for i, (lbl, w) in enumerate(items):
        bottom = 0.0
        for g in REF:
            v = w["p_" + g]
            ax.bar(i, v, 0.7, bottom=bottom, color=pg.GAIT_COLORS[g], edgecolor="white", lw=0.6,
                   label="mode " + pg.LABELS[g] if i == 0 else None)
            bottom += v
        ax.text(i, 1.03, "{:.0%}".format(w["off_recon"]), ha="center", va="bottom", fontsize=8, color="#6f6b63")
    ax.text(-0.6, 1.11, "reconstruction off-manifold:", fontsize=8, color="#6f6b63")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([l for l, _ in items], fontsize=9)
    ax.set_ylabel("fraction of windows (all)")
    ax.set_ylim(0, 1.18)
    ax.set_title("Modes used by the policies (VQ-VAE codes)", loc="left", color=pg.INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "vqvae_modes_policies.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    keys = ["location", "steering", "run-only"] + ["one-hot, c = " + g for g in gnames]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    ref_pc, ref_g = ck["ref_hist_pc"], np.asarray(ck["ref_gait"])
    for ax, key in zip(axs.ravel(), keys):
        for g in range(len(REF)):
            m = ref_g == g
            ax.scatter(ref_pc[m, 0], ref_pc[m, 1], s=8, color=pg.GAIT_COLORS[REF[g]], alpha=0.35, lw=0,
                       label="MoCap " + pg.LABELS[REF[g]] if key == keys[0] else None)
        p, off = pcs[key]
        off = off[1]
        sel = np.random.default_rng(0).permutation(len(p))[:3000]
        p, off = p[sel], off[sel]
        ax.scatter(p[~off, 0], p[~off, 1], s=4, color=pg.INK, alpha=0.35, lw=0, label="policy" if key == keys[0] else None)
        ax.scatter(p[off, 0], p[off, 1], s=4, color="#9b978f", alpha=0.35, lw=0,
                   label="policy, reconstruction off-manifold" if key == keys[0] else None)
        ax.set_title(key, loc="left", color=pg.INK, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=5,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(args.out_dir, "vqvae_policy_pca.png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("Saved figures to", args.out_dir)


if __name__ == "__main__":
    main()
