"""
Morphology-agnostic gait analysis of the reference MoCap in the AMP discriminator's own feature space.

No contact thresholds, foot rules or gait labels are used to build the representation or the clusters:
  feature  = the AMP discriminator observation of a window (root height and rotation, local root linear
             and angular velocity, joint rotations and velocities, key-body positions in the root frame),
             normalized by the MoCap mean / std
  short    = 10 steps (0.33 s), the window the AMP discriminator sees (num_disc_obs_steps)
  long     = 30 steps (1 s), at least one full stride of every gait
  weights  = k-means weights each window by its clip's dataset sampling weight, so the 4 long walk clips
             do not outweigh the short trot and run clips (same distribution the AMP demo sampler uses)
  micro    = k-means (K_micro) on short windows: motion primitives, one per phase / posture
  macro    = k-means on the histogram of micro codes inside each long window ("bag of motion words",
             the same aggregation as the modal code of a VQ-VAE window); also k-means directly on the
             PCA-reduced long windows for comparison

Gait names from the clip files, and the footfall classifier of gait_collapse.py, are used only to
evaluate the representation (silhouette, kNN accuracy) and to name the clusters (purity, NMI, ARI).

Outputs in --out_dir: amp_feature_clusters.txt, amp_feature_pca.png, amp_feature_clusters.png,
amp_feature_clusters.npz

Usage (from MimicKit/):
  python tools/gait_analysis/amp_feature_clusters.py --run_dir output/go2_steer_amp_allgaits_2048env_s42 \
      --out_dir output/figures/amp_features
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

REF = gc.REF_GAITS            # walk_pace, trot, run


#####################################################################
# numerics (no scikit-learn in the env)
#####################################################################

def pca(x, k):
    mu = x.mean(0, keepdim=True)
    u, s, vt = torch.linalg.svd(x - mu, full_matrices=False)
    var = s ** 2
    return (x - mu) @ vt[:k].T, (var[:k] / var.sum()).cpu().numpy()


def kmeans(x, k, w=None, n_init=10, iters=100, seed=0):
    """Weighted k-means (k-means++ init). w: sample weights (None = uniform)."""
    w = torch.ones(len(x), device=x.device) if w is None else w
    g = torch.Generator(device=x.device).manual_seed(seed)
    best, best_in = None, float("inf")
    for _ in range(n_init):
        c = x[torch.multinomial(w / w.sum(), 1, generator=g)]
        for _ in range(1, k):
            d = w * torch.cdist(x, c).min(1).values ** 2
            c = torch.cat([c, x[torch.multinomial(d / d.sum(), 1, generator=g)]])
        for _ in range(iters):
            a = torch.cdist(x, c).argmin(1)
            new = torch.stack([(w[a == j, None] * x[a == j]).sum(0) / w[a == j].sum() if (a == j).any() else c[j]
                               for j in range(k)])
            if torch.allclose(new, c):
                break
            c = new
        inertia = (w * torch.cdist(x, c).min(1).values ** 2).sum().item()
        if inertia < best_in:
            best, best_in = (a.clone(), c.clone()), inertia
    return best[0], best[1]


def silhouette(x, lab):
    d = torch.cdist(x, x)
    lab = torch.as_tensor(lab, device=x.device)
    ks = lab.unique()
    if len(ks) < 2:
        return float("nan")
    mean_to = torch.stack([d[:, lab == k].mean(1) for k in ks], 1)          # [N, K]
    own = (lab[:, None] == ks[None, :])
    n_own = own.float() @ torch.stack([(lab == k).sum() for k in ks]).float()
    a = (mean_to * own).sum(1) * n_own / (n_own - 1).clamp_min(1)          # exclude self
    b = mean_to.masked_fill(own, float("inf")).min(1).values
    return ((b - a) / torch.maximum(a, b)).mean().item()


def contingency(a, b, na, nb):
    m = np.zeros((na, nb))
    np.add.at(m, (a, b), 1)
    return m


def nmi(a, b):
    m = contingency(a, b, a.max() + 1, b.max() + 1) / len(a)
    pa, pb = m.sum(1), m.sum(0)
    nz = m > 0
    mi = (m[nz] * np.log(m[nz] / (pa[:, None] * pb[None, :])[nz])).sum()
    h = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    return mi / max(np.sqrt(h(pa) * h(pb)), 1e-12)


def ari(a, b):
    m = contingency(a, b, a.max() + 1, b.max() + 1)
    c2 = lambda x: x * (x - 1) / 2
    s, sa, sb, n = c2(m).sum(), c2(m.sum(1)).sum(), c2(m.sum(0)).sum(), c2(len(a))
    exp = sa * sb / n
    return (s - exp) / max(0.5 * (sa + sb) - exp, 1e-12)


def purity(clu, lab):
    m = contingency(clu, lab, clu.max() + 1, lab.max() + 1)
    return m.max(1).sum() / m.sum()


def knn_acc(x, lab, clip, start, win_sec, k=5):
    """Gait label of the k nearest windows, excluding overlapping windows of the same clip."""
    d = torch.cdist(x, x)
    same = (clip[:, None] == clip[None, :]) & ((start[:, None] - start[None, :]).abs() < win_sec)
    d = d.masked_fill(same, float("inf"))
    idx = d.topk(k, largest=False).indices
    pred = torch.mode(lab[idx], dim=1).values
    return (pred == lab).float().mean().item()


#####################################################################
# AMP features of the reference
#####################################################################

def amp_windows(env, clip_ids, t_end, steps):
    """AMP discriminator observation of the windows of `steps` frames ending at t_end."""
    old = env._num_disc_obs_steps
    env._num_disc_obs_steps = steps
    out = []
    for i in range(0, len(t_end), 4096):
        out.append(env._compute_disc_obs_demo(clip_ids[i:i + 4096], t_end[i:i + 4096]))
    env._num_disc_obs_steps = old
    return torch.cat(out)


def standardize(x):
    mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True)
    return (x - mu) / sd.clamp_min(1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="any run whose env_config holds the all-gaits dataset")
    ap.add_argument("--out_dir", default="output/figures/amp_features")
    ap.add_argument("--short_steps", type=int, default=10)
    ap.add_argument("--long_steps", type=int, default=30)
    ap.add_argument("--k_micro", type=int, default=32)
    ap.add_argument("--k_macro", default="2,3,4,5,6")
    ap.add_argument("--pcs", type=int, default=16, help="PCs kept for clustering long windows directly")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(0)

    mp_util.init(0, 1, args.device, None)
    env = env_builder.build_env(os.path.join(args.run_dir, "env_config.yaml"),
                                os.path.join(args.run_dir, "engine_config.yaml"), 1, args.device, visualize=False)
    dt = env._engine.get_timestep()
    nd = env._engine.get_dof_pos(env._get_char_id()).shape[-1]
    L, S = args.long_steps, args.short_steps

    # reference long windows (same windows and footfall labels as gait_collapse.py)
    ref = gc.build_reference(env, L)
    clip, start = ref["clip_ids"], ref["starts"]
    file_gait = ref["labels"]                                         # gait from the clip file name
    st = gc.footfall_stats(ref["feats"], nd, gc.resting_foot_height(ref["feats"], nd))
    foot_gait = gc.footfall_classify(st)                              # gait from the footfall rule
    t_end = start + (L - 1) * dt
    n = len(clip)
    clip_w = env._motion_lib.get_motion_weights()
    win_w = clip_w[clip] / torch.bincount(clip, minlength=len(clip_w)).float()[clip]
    win_w = win_w / win_w.mean()                                      # dataset sampling distribution

    long_x = standardize(amp_windows(env, clip, t_end, L))
    short_x = standardize(amp_windows(env, clip, t_end, S))           # the last 0.33 s of each long window

    # micro windows: every short window that fits inside each long window
    offs = torch.arange(S - 1, L, device=clip.device) * dt            # end offsets from the window start
    m_clip = clip[:, None].expand(-1, len(offs)).reshape(-1)
    m_end = (start[:, None] + offs[None, :]).reshape(-1)
    micro_raw = amp_windows(env, m_clip, m_end, S)
    micro_x = (micro_raw - micro_raw.mean(0)) / micro_raw.std(0).clamp_min(1e-3)
    micro_code, _ = kmeans(micro_x, args.k_micro, w=win_w[:, None].expand(-1, len(offs)).reshape(-1), n_init=4)
    hist = torch.zeros(n, args.k_micro, device=clip.device)
    hist.scatter_add_(1, micro_code.view(n, -1), torch.ones(n, len(offs), device=clip.device))
    bow = torch.sqrt(hist / hist.sum(1, keepdim=True))                # Hellinger embedding of the histogram
    micro_used = (torch.bincount(micro_code, minlength=args.k_micro) > 0).sum().item()

    # representation quality, no clustering
    fg, gl = file_gait.cpu().numpy(), foot_gait.cpu().numpy()
    names = ref["clip_names"]
    lines = ["Reference MoCap: {} windows of {} steps ({:.2f} s), {} clips".format(n, L, L * dt, len(names)),
             "Features: AMP discriminator observation, standardized; short = {} steps ({:.2f} s), long = {} steps ({:.2f} s)".format(
                 S, S * dt, L, L * dt),
             "Gait labels used only for evaluation: file name (walk_pace / trot / run) and footfall rule",
             "k-means weights each window by the dataset sampling weight of its clip (as the AMP demo sampler);",
             "windows per file gait: " + ", ".join("{}={}".format(g, int((file_gait == i).sum())) for i, g in enumerate(REF)), ""]
    reps = {"short (AMP window)": short_x, "long (1 s)": long_x, "micro-code histogram": bow}
    lines.append("Representation           dims  PC1+PC2 var  silhouette(file gait)  kNN acc file  kNN acc footfall")
    pcs = {}
    for name, x in reps.items():
        p2, var = pca(x, 2)
        pcs[name] = p2.cpu().numpy()
        sil = silhouette(x, file_gait)
        ka = knn_acc(x, file_gait, clip, start, L * dt)
        kf = knn_acc(x, foot_gait, clip, start, L * dt)
        lines.append("{:24s} {:5d}  {:10.0%}  {:21.2f}  {:12.0%}  {:16.0%}".format(name, x.shape[1], var.sum(), sil, ka, kf))
    lines.append("")

    # macro clusters
    long_red, _ = pca(long_x, args.pcs)
    spaces = {"micro-code histogram": bow, "long, {} PCs".format(args.pcs): long_red}
    ks = [int(k) for k in args.k_macro.split(",")]
    lines.append("Macro clustering (k-means)   K  silhouette  NMI/ARI/purity vs file gait  NMI/ARI/purity vs footfall")
    macro, sil_curve = {}, {}
    for sname, x in spaces.items():
        sil_curve[sname] = []
        for k in ks:
            lab, _ = kmeans(x, k, w=win_w)
            lab = lab.cpu().numpy()
            s = silhouette(x, lab)
            sil_curve[sname].append(s)
            macro[(sname, k)] = lab
            lines.append("{:28s} {:2d}  {:10.2f}  {:.2f} / {:.2f} / {:.0%}            {:.2f} / {:.2f} / {:.0%}".format(
                sname, k, s, nmi(lab, fg), ari(lab, fg), purity(lab, fg), nmi(lab, gl), ari(lab, gl), purity(lab, gl)))
    lines.append("")

    # K = 3 on the micro-code histogram: contingency, and where the trot segments of the walk clips go
    main_space = "micro-code histogram"
    lab3 = macro[(main_space, 3)]
    cf = contingency(lab3, fg, 3, len(REF))
    cl = contingency(lab3, gl, 3, len(gc.GAITS))
    lines.append("K = 3 on the {} (rows = cluster, micro codes used: {}/{}):".format(main_space, micro_used, args.k_micro))
    lines.append("  vs file gait     " + "  ".join("{:>9s}".format(g) for g in REF))
    for i in range(3):
        lines.append("  cluster {}        ".format(i) + "  ".join("{:9d}".format(int(v)) for v in cf[i]))
    lines.append("  vs footfall gait " + "  ".join("{:>9s}".format(g) for g in gc.GAITS))
    for i in range(3):
        lines.append("  cluster {}        ".format(i) + "  ".join("{:9d}".format(int(v)) for v in cl[i]))
    trot_cluster = int(np.argmax(cf[:, REF.index("trot")]))
    walk_trot = (fg == REF.index("walk_pace")) & (gl == gc.GAITS.index("trot"))
    if walk_trot.any():
        lines.append("  windows of walk clips that the footfall rule calls trot: {}; in the cluster of the trot clip: {:.0%}".format(
            walk_trot.sum(), (lab3[walk_trot] == trot_cluster).mean()))
    walk_walk = (fg == REF.index("walk_pace")) & (gl == gc.GAITS.index("walk_pace"))
    lines.append("  windows of walk clips that the footfall rule calls walk: {}; in the cluster of the trot clip: {:.0%}".format(
        walk_walk.sum(), (lab3[walk_walk] == trot_cluster).mean()))

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out_dir, "amp_feature_clusters.txt"), "w") as f:
        f.write(report + "\n")
    np.savez(os.path.join(args.out_dir, "amp_feature_clusters.npz"), clip=clip.cpu().numpy(), start=start.cpu().numpy(),
             clip_names=np.array(names), file_gait=fg, foot_gait=gl, speed=st["speed"].cpu().numpy(),
             **{"pca_" + k.split(" ")[0]: v for k, v in pcs.items()}, macro3=lab3, micro_hist=hist.cpu().numpy())

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_gait_collapse as pg

    titles = {"short (AMP window)": "AMP window, {:.2f} s".format(S * dt), "long (1 s)": "long window, {:.0f} s".format(L * dt),
              "micro-code histogram": "histogram of {} micro codes, 1 s".format(args.k_micro)}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (name, p) in zip(axs, pcs.items()):
        for g in range(len(REF)):
            m = fg == g
            ax.scatter(p[m, 0], p[m, 1], s=7, color=pg.GAIT_COLORS[REF[g]], alpha=0.6, lw=0, label=pg.LABELS[REF[g]])
        ax.set_title(titles[name], loc="left", color=pg.INK, fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xticks([])
        ax.set_yticks([])
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False, fontsize=9)
    fig.savefig(os.path.join(args.out_dir, "amp_feature_pca.png"), bbox_inches="tight", dpi=200)

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    p = pcs["micro-code histogram"]
    ccol = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#eda100", "#e87ba4"]
    for i in range(3):
        m = lab3 == i
        axs[0].scatter(p[m, 0], p[m, 1], s=7, color=ccol[i], alpha=0.6, lw=0, label="cluster {}".format(i))
    axs[0].set_title("K = 3 clusters, no labels used", loc="left", color=pg.INK, fontsize=11)
    axs[0].set_xticks([]); axs[0].set_yticks([])
    axs[0].set_xlabel("PC1"); axs[0].set_ylabel("PC2")
    axs[0].legend(frameon=False, fontsize=9)
    for ax, mat, cols, ttl in [(axs[1], cf, REF, "clusters vs clip file")]:
        frac = mat / mat.sum(0, keepdims=True).clip(min=1)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, "{:.0%}\n({})".format(frac[i, j], int(mat[i, j])), ha="center", va="center", fontsize=8,
                        color="white" if frac[i, j] > 0.6 else pg.INK)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([pg.LABELS[c] + " clips" for c in cols])
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels(["cluster {}".format(i) for i in range(mat.shape[0])])
        ax.set_title(ttl + " (column %)", loc="left", color=pg.INK, fontsize=11)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "amp_feature_clusters.png"), bbox_inches="tight", dpi=200)

    # micro codes on the phase rings, and the bag of motion words of a few real 1 s windows
    micro_fig(args, micro_x, micro_code, win_w, clip, start, fg, names, n, len(offs), dt, pg, plt)
    print("Saved figures to", args.out_dir)


def micro_fig(args, micro_x, micro_code, win_w, clip, start, fg, names, n, m, dt, pg, plt):
    """amp_feature_micro.png: short windows in PCA coloured by micro code (one panel per gait, shared
    axes), the path of one 1 s walk window along its ring, and the code strips + histograms of four
    real 1 s windows (two walk windows at different phases, one trot, one run)."""
    import matplotlib.colors as mcolors
    K = args.k_micro
    codes = micro_code.cpu().numpy()
    gait_of_short = np.repeat(fg, m)
    w_short = np.repeat(win_w.cpu().numpy(), m)
    # dominant gait of each code (weighted), and a colour family per gait
    mass = np.zeros((K, len(REF)))
    np.add.at(mass, (codes, gait_of_short), w_short)
    dom = mass.argmax(1)
    cmaps = {"walk_pace": "Blues", "trot": "Oranges", "run": "Greens"}
    p2, _ = pca(micro_x, 2)
    p2 = p2.cpu().numpy()
    # within each gait, order the codes by their angle on the gait's ring, so colour and number follow the phase
    colors = np.zeros((K, 4))
    order = []
    for g, gname in enumerate(REF):
        ks = [k for k in range(K) if dom[k] == g]
        ctr = p2[gait_of_short == g].mean(0)
        ang = [np.arctan2(*(p2[codes == k].mean(0) - ctr)[::-1]) for k in ks]
        ks = [k for _, k in sorted(zip(ang, ks))]
        order += ks
        cm = plt.get_cmap(cmaps[gname])
        for i, k in enumerate(ks):
            colors[k] = cm(0.3 + 0.7 * (i / max(len(ks) - 1, 1)))
    start_np, clip_np = start.cpu().numpy(), clip.cpu().numpy()

    def win_index(name, t):
        c = names.index(name)
        idx = np.nonzero(clip_np == c)[0]
        return int(idx[np.argmin(np.abs(start_np[idx] - t))])

    walk_clip = [nm for nm in names if gc.GAIT_OF_CLIP[nm] == "walk_pace"][0]
    trot_clip = [nm for nm in names if gc.GAIT_OF_CLIP[nm] == "trot"][0]
    run_clip = [nm for nm in names if gc.GAIT_OF_CLIP[nm] == "run"][0]
    examples = [("walk ({}, t = 2.0 s)".format(walk_clip), win_index(walk_clip, 2.0)),
                ("walk ({}, t = 2.5 s)".format(walk_clip), win_index(walk_clip, 2.5)),
                ("trot ({}, t = 0.0 s)".format(trot_clip), win_index(trot_clip, 0.0)),
                ("run ({}, t = 1.0 s)".format(run_clip), win_index(run_clip, 1.0))]

    fig = plt.figure(figsize=(15, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.28, wspace=0.12)
    lim_x = (p2[:, 0].min() - 1, p2[:, 0].max() + 1)
    lim_y = (p2[:, 1].min() - 1, p2[:, 1].max() + 1)
    for g, gname in enumerate(REF):
        ax = fig.add_subplot(gs[0, g])
        sel = gait_of_short == g
        ax.scatter(p2[~sel, 0], p2[~sel, 1], s=3, color="#d9d7d0", lw=0, alpha=0.5)
        ax.scatter(p2[sel, 0], p2[sel, 1], s=6, c=colors[codes[sel]], lw=0, alpha=0.85)
        if g == 0:
            i0 = examples[0][1]
            path = p2[i0 * m:(i0 + 1) * m]
            ax.plot(path[:, 0], path[:, 1], "-", color=pg.INK, lw=1.2)
            ax.scatter(path[0, 0], path[0, 1], s=30, color=pg.INK, zorder=5)
            ax.annotate("one 1 s window\n(21 short windows)", path[0], xytext=(10, -28), textcoords="offset points",
                        fontsize=8, color=pg.INK)
        ax.set_xlim(*lim_x)
        ax.set_ylim(*lim_y)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        nk = int((dom == g).sum())
        ax.set_title("{} clips: 0.33 s windows, colour = micro code\n({} codes mostly {})".format(
            pg.LABELS[gname], nk, pg.LABELS[gname]), loc="left", color=pg.INK, fontsize=10)

    sub = gs[1, :].subgridspec(len(examples), 2, width_ratios=[1.25, 1], hspace=0.65, wspace=0.08)
    rank = {k: i for i, k in enumerate(order)}
    hmax = 1.1 * max(np.bincount(codes[i * m:(i + 1) * m], minlength=K).max() for _, i in examples)
    for r, (lbl, i) in enumerate(examples):
        cs = codes[i * m:(i + 1) * m]
        ax = fig.add_subplot(sub[r, 0])
        for j, k in enumerate(cs):
            ax.add_patch(plt.Rectangle((j, 0), 0.92, 1, color=colors[k]))
            ax.text(j + 0.46, 0.5, str(rank[k]), ha="center", va="center", fontsize=6.5,
                    color="white" if mcolors.rgb_to_hsv(colors[k][:3])[2] < 0.75 else pg.INK)
        ax.set_xlim(0, m)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(lbl + ": micro code of each 0.33 s window, in time order", loc="left", color=pg.INK, fontsize=9)
        ax = fig.add_subplot(sub[r, 1])
        h = np.bincount(cs, minlength=K)[order]
        ax.bar(np.arange(K), h, color=colors[order], width=0.85)
        ax.set_xlim(-0.6, K - 0.4)
        ax.set_ylim(0, hmax)
        ax.set_yticks([])
        ax.set_xticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        if r == 0:
            ax.set_title("histogram over the {} codes (numbered by gait, then by phase)".format(K), loc="left", color=pg.INK, fontsize=9)
    fig.savefig(os.path.join(args.out_dir, "amp_feature_micro.png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
