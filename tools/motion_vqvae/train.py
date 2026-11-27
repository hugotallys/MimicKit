"""
Train the motion VQ-VAE on the MoCap, in the AMP discriminator feature space, and evaluate its codes as a
gait representation (offline; no policy, no simulation step, MimicKit untouched).

Data     random W-frame MoCap windows, clips drawn with the dataset weights (as the AMP demo sampler),
         per-frame AMP features standardized by the MoCap mean / std per channel.
Model    model.MotionVQVAE (EMA codebook, dead-code reset); W = 32 frames (1.07 s at 30 Hz), 2^n_down
         temporal downsampling -> W / 2^n_down tokens per window. The pace and trot clips are single
         looping cycles of ~0.6 s and the run clip lasts 3.4 s, so windows of 128 frames (4.3 s) do not fit
         the dataset; W = 32 covers at least one stride of every gait.
         Defaults K = 16, n_down = 1 (16 tokens of 2 frames) come from a sweep over K in {16, 32, 64} and
         n_down in {1, 2, 3}: the modal code never separates the gaits (ARI <= 0.04 in all nine), the token
         histogram does when a window has many tokens (ARI ~0.7 with 16 tokens, fails with 4), and this
         setting separates the K = 3 histogram clusters best (silhouette 0.34).
Readouts on the same 1057 reference windows as amp_feature_clusters.py (window end = end of each 1 s
         window there), labels used only to evaluate:
           modal code       the most frequent token of the window (the interface proposed for the policy)
           token histogram  Hellinger histogram of the window's tokens -> weighted k-means (K = 2..6),
                            the macro modes; named by majority clip
           pooled z_e       mean of the continuous encoder output over the window
         Off-manifold threshold for policies: 1.5 x the MoCap 95th percentile of the window's
         reconstruction MSE. (The latent token-to-code distance is also saved, but the encoder pulls
         unseen motion close to some code, so it under-detects off-manifold windows.)

Outputs  --ckpt (model, normalization, macro centroids, names, threshold) and in --out_dir:
         vqvae_mocap.txt, vqvae_pca.png, vqvae_codes.png, vqvae_training.png, vqvae_mocap.npz

Usage (from MimicKit/):
  python tools/motion_vqvae/train.py --run_dir output/go2_steer_amp_allgaits_2048env_s42
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "gait_analysis"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "mimickit"))

import amp_feature_clusters as afc  # noqa: E402
import features as feat  # noqa: E402
import gait_collapse as gc  # noqa: E402
import util.mp_util as mp_util  # noqa: E402
from model import MotionVQVAE  # noqa: E402

REF = gc.REF_GAITS
L_REF = 30          # reference windows of amp_feature_clusters.py
OFF_FACTOR = 1.5


def hellinger_hist(codes, k):
    h = torch.zeros(codes.shape[0], k, device=codes.device)
    h.scatter_add_(1, codes, torch.ones_like(codes, dtype=torch.float))
    return torch.sqrt(h / h.sum(1, keepdim=True))


def perplexity(codes, k):
    p = torch.bincount(codes.flatten(), minlength=k).float()
    p = p / p.sum()
    return float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="output/go2_steer_amp_allgaits_2048env_s42",
                    help="any run whose env_config holds the all-gaits dataset")
    ap.add_argument("--out_dir", default="output/figures/motion_vqvae")
    ap.add_argument("--ckpt", default="output/motion_vqvae/vqvae_go2_allgaits.pt")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--n_down", type=int, default=1)
    ap.add_argument("--codes", type=int, default=16)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--k_macro", default="2,3,4,5,6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mp_util.init(0, 1, args.device, None)
    env = feat.build_env(args.run_dir, 1, args.device)
    dt = env._engine.get_timestep()
    W = args.window
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    # normalization from a large weighted sample of MoCap windows
    big, _ = feat.sample_mocap(env, 20000, W, gen)
    C = big.shape[-1]
    mu, sd = big.reshape(-1, C).mean(0), big.reshape(-1, C).std(0).clamp_min(1e-3)
    norm = lambda x: ((x - mu) / sd).permute(0, 2, 1).contiguous()        # (N, W, C) -> (N, C, W)

    model = MotionVQVAE(C, args.latent, args.codes, args.hidden, args.n_down).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * 0.1)
    log = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, _ = feat.sample_mocap(env, args.batch, W, gen)
        x = norm(x)
        model.train()
        recon, commit, idx, active, n_dead = model(x)
        rec = F.mse_loss(recon, x)
        loss = rec + commit
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1:
            log.append((step, rec.item(), commit.item(), active, perplexity(idx, args.codes), n_dead))
            print("step {:5d}  recon {:.4f}  commit {:.4f}  active {:2d}/{}  perplexity {:5.1f}  dead reset {:2d}  ({:.0f} s)".format(
                *log[-1][:4], args.codes, *log[-1][4:], time.time() - t0))

    # ---------------- evaluation on the reference windows
    model.eval()
    ref = gc.build_reference(env, L_REF)
    clip, start = ref["clip_ids"], ref["starts"]
    fg = ref["labels"].cpu().numpy()
    n = len(clip)
    t_end = start + (L_REF - 1) * dt
    clip_w = env._motion_lib.get_motion_weights()
    win_w = clip_w[clip] / torch.bincount(clip, minlength=len(clip_w)).float()[clip]
    win_w = win_w / win_w.mean()

    xr = norm(feat.mocap_windows(env, clip, t_end, W))
    with torch.no_grad():
        recon, _, _, _, _ = model(xr)
        rec_mse = F.mse_loss(recon, xr).item()
        codes, z_e, qdist = model.encode(xr)
    T = codes.shape[1]
    modal = model.modal_code(codes)
    hist = hellinger_hist(codes, args.codes)
    pooled = z_e.mean(2)
    pooled = (pooled - pooled.mean(0)) / pooled.std(0).clamp_min(1e-6)
    off_stat = qdist.median(1).values
    off_thr = OFF_FACTOR * torch.quantile(off_stat, 0.95).item()
    rec_err = (recon - xr).pow(2).mean((1, 2))
    off_thr_rec = OFF_FACTOR * torch.quantile(rec_err, 0.95).item()

    lines = ["Motion VQ-VAE on the MoCap, AMP discriminator features ({} channels per frame)".format(C),
             "window W = {} frames ({:.2f} s), {} tokens per window ({} frames each), K = {} codes, D = {}".format(
                 W, W * dt, T, 2 ** args.n_down, args.codes, args.latent),
             "training: {} steps x {} windows (clips drawn with the dataset weights)".format(args.steps, args.batch),
             "reference: {} windows (the 1 s windows of amp_feature_clusters.py, VQ window ending at the same frame)".format(n),
             "labels (clip file gait) used only for evaluation", "",
             "recon MSE (standardized units): {:.4f}".format(rec_mse),
             "codes used on the MoCap: {}/{}   perplexity: {:.1f}".format(
                 int((torch.bincount(codes.flatten(), minlength=args.codes) > 0).sum()), args.codes,
                 perplexity(codes, args.codes))]
    used = {}
    for g in range(len(REF)):
        m = torch.as_tensor(fg == g, device=codes.device)
        used[g] = set(torch.unique(codes[m]).tolist())
        lines.append("  codes used by {:9s} windows: {:2d}".format(REF[g], len(used[g])))
    for a in range(len(REF)):
        for b in range(a + 1, len(REF)):
            lines.append("  shared {} / {}: {}".format(REF[a], REF[b], len(used[a] & used[b])))
    lines.append("")

    # representations
    reps = {"pooled z_e": pooled, "token histogram": hist}
    lines.append("Representation        dims  PC1+PC2 var  silhouette(file gait)  kNN acc file")
    pcs = {}
    for name, x in reps.items():
        p2, var = afc.pca(x, 2)
        pcs[name] = p2.cpu().numpy()
        lines.append("{:20s} {:5d}  {:10.0%}  {:21.2f}  {:12.0%}".format(
            name, x.shape[1], var.sum(), afc.silhouette(x, ref["labels"]),
            afc.knn_acc(x, ref["labels"], clip, start, L_REF * dt)))
    lines.append("")

    mod = modal.cpu().numpy()
    _, mod_c = np.unique(mod, return_inverse=True)
    lines.append("Modal code as the cluster: {} distinct modal codes; NMI {:.2f}  ARI {:.2f}  purity {:.0%} vs file gait".format(
        mod_c.max() + 1, afc.nmi(mod_c, fg), afc.ari(mod_c, fg), afc.purity(mod_c, fg)))
    lines.append("")

    ks = [int(k) for k in args.k_macro.split(",")]
    lines.append("Macro clustering (weighted k-means)   K  silhouette  NMI / ARI / purity vs file gait")
    macro, sil = {}, {}
    for name, x in reps.items():
        sil[name] = []
        for k in ks:
            lab, cent = afc.kmeans(x, k, w=win_w, seed=args.seed)
            lab_np = lab.cpu().numpy()
            s = afc.silhouette(x, lab_np)
            sil[name].append(s)
            macro[(name, k)] = (lab_np, cent)
            lines.append("{:36s} {:2d}  {:10.2f}  {:.2f} / {:.2f} / {:.0%}".format(
                name, k, s, afc.nmi(lab_np, fg), afc.ari(lab_np, fg), afc.purity(lab_np, fg)))
    lines.append("")

    lab3, mcent = macro[("token histogram", 3)]
    cf = afc.contingency(lab3, fg, 3, len(REF))
    names = [REF[int(np.argmax(cf[i] / cf.sum(0).clip(min=1)))] for i in range(3)]
    lines.append("K = 3 on the token histogram (rows = cluster; named by majority clip: {}):".format(names))
    lines.append("  vs file gait   " + "  ".join("{:>9s}".format(g) for g in REF))
    for i in range(3):
        lines.append("  cluster {}      ".format(i) + "  ".join("{:9d}".format(int(v)) for v in cf[i]))
    q = np.array([win_w[torch.as_tensor(lab3 == names.index(g), device=win_w.device)].sum().item()
                  if g in names else 0.0 for g in REF])
    lines.append("MoCap mode distribution (dataset weights): " + ", ".join(
        "{}={:.2f}".format(g, v) for g, v in zip(REF, q / q.sum())))
    lines.append("off-manifold thresholds: reconstruction MSE {:.4f} (used), median token-to-code distance {:.3f}".format(
        off_thr_rec, off_thr))

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out_dir, "vqvae_mocap.txt"), "w") as f:
        f.write(report + "\n")

    hmu = hist.mean(0, keepdim=True)
    _, _, vt = torch.linalg.svd(hist - hmu, full_matrices=False)
    torch.save({"state_dict": model.state_dict(), "config": model.config, "window": W, "mu": mu, "sd": sd,
                "macro_centroids": mcent, "macro_names": names, "off_thr": off_thr, "off_thr_rec": off_thr_rec, "ref_dist": q / q.sum(),
                "hist_mu": hmu, "hist_basis": vt[:2].T, "ref_hist_pc": ((hist - hmu) @ vt[:2].T).cpu().numpy(),
                "ref_gait": fg, "run_dir": args.run_dir}, args.ckpt)
    np.savez(os.path.join(args.out_dir, "vqvae_mocap.npz"), clip=clip.cpu().numpy(), start=start.cpu().numpy(),
             file_gait=fg, codes=codes.cpu().numpy(), modal=mod, macro3=lab3, log=np.array(log),
             **{"pca_" + k.replace(" ", "_"): v for k, v in pcs.items()})
    print("Saved", args.ckpt)

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_gait_collapse as pg

    # 1) PCA: pooled z_e, token histogram (gait colours), K = 3 clusters
    ccol = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, name in zip(axs[:2], ["pooled z_e", "token histogram"]):
        p = pcs[name]
        for g in range(len(REF)):
            m = fg == g
            ax.scatter(p[m, 0], p[m, 1], s=7, color=pg.GAIT_COLORS[REF[g]], alpha=0.6, lw=0, label=pg.LABELS[REF[g]] + " clips")
    axs[0].set_title("VQ-VAE encoder output, mean over the window", loc="left", color=pg.INK, fontsize=11)
    axs[1].set_title("histogram of the window's {} tokens".format(T), loc="left", color=pg.INK, fontsize=11)
    p = pcs["token histogram"]
    for i in range(3):
        m = lab3 == i
        axs[2].scatter(p[m, 0], p[m, 1], s=7, color=ccol[REF.index(names[i])], alpha=0.6, lw=0,
                       label="cluster {} ({})".format(i, pg.LABELS[names[i]]))
    axs[2].set_title("K = 3 clusters of the histogram, no labels used", loc="left", color=pg.INK, fontsize=11)
    axs[2].legend(frameon=False, fontsize=8)
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False, fontsize=9)
    fig.savefig(os.path.join(args.out_dir, "vqvae_pca.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    # 2) code usage per gait, and the token sequence of one clip per gait
    cnp = codes.cpu().numpy()
    usage = np.zeros((len(REF), args.codes))
    for g in range(len(REF)):
        usage[g] = np.bincount(cnp[fg == g].ravel(), minlength=args.codes)
        usage[g] /= max(usage[g].sum(), 1)
    order = np.lexsort((-usage.max(0), usage.argmax(0)))
    order = order[usage[:, order].sum(0) > 0]
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.0), gridspec_kw={"width_ratios": [1.3, 1]})
    axs[0].imshow(usage[:, order], aspect="auto", cmap="Blues", vmin=0, vmax=usage.max())
    axs[0].set_yticks(range(len(REF)))
    axs[0].set_yticklabels([pg.LABELS[g] + " clips" for g in REF])
    axs[0].set_xticks(range(len(order)))
    axs[0].set_xticklabels([str(k) for k in order], fontsize=6, rotation=90)
    axs[0].set_xlabel("code (sorted by the gait that uses it most)")
    axs[0].set_title("p(code | gait of the clip)", loc="left", color=pg.INK, fontsize=11)
    axs[0].grid(False)
    rank = {k: i for i, k in enumerate(order)}
    lib = env._motion_lib
    names_clip = ref["clip_names"]
    for g in range(len(REF)):
        cands = [m for m in range(len(names_clip)) if gc.GAIT_OF_CLIP[names_clip[m]] == REF[g]]
        m = cands[0]
        loop = lib.get_motion_loop_mode(torch.tensor([m], device=clip.device)).item() == 1
        t_hi = (W - 1) * dt + 3.0 if loop else min(lib.get_motion_lengths()[m].item(), (W - 1) * dt + 3.0)
        tt = torch.arange((W - 1) * dt, t_hi, dt, device=clip.device)
        with torch.no_grad():
            cs, _, _ = model.encode(norm(feat.mocap_windows(env, torch.full_like(tt, m, dtype=torch.long), tt, W)))
        y = [rank.get(int(k), -1) for k in cs[:, -1].cpu().numpy()]
        axs[1].plot(tt.cpu().numpy(), y, "-o", ms=3, lw=0.8, color=pg.GAIT_COLORS[REF[g]],
                    label="{} ({})".format(pg.LABELS[REF[g]], names_clip[m]))
    axs[1].set_xlabel("time in the clip (s)")
    axs[1].set_ylabel("code of the newest token (sorted as left)")
    axs[1].set_title("newest token over time, one clip per gait", loc="left", color=pg.INK, fontsize=11)
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "vqvae_codes.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    # 3) training curves
    lg = np.array(log)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.2))
    axs[0].plot(lg[:, 0], lg[:, 1], color=pg.INK)
    axs[0].set_yscale("log")
    axs[0].set_title("reconstruction MSE", loc="left", color=pg.INK, fontsize=11)
    axs[1].plot(lg[:, 0], lg[:, 3], color="#2a78d6", label="active in batch")
    axs[1].plot(lg[:, 0], lg[:, 4], color="#eb6834", label="perplexity")
    axs[1].set_title("codebook use (K = {})".format(args.codes), loc="left", color=pg.INK, fontsize=11)
    axs[1].legend(frameon=False, fontsize=8)
    axs[2].plot(lg[:, 0], lg[:, 5], color="#8a5cd1")
    axs[2].set_title("dead codes reset per step", loc="left", color=pg.INK, fontsize=11)
    for ax in axs:
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "vqvae_training.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Saved figures to", args.out_dir)


if __name__ == "__main__":
    main()
