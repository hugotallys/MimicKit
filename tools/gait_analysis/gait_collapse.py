"""
Gait mode-collapse analysis for AMP policies trained on a multi-gait dataset.

For each intermediate checkpoint of a run, the policy is rolled out with deterministic actions
and every 1-second window of motion is assigned a gait class. The gait distribution of the
policy is compared with the reference distribution defined by the dataset weights.

Gait classification (two independent methods):
  footfall  (primary)   rule based on which feet touch the ground together:
                        trot = diagonal pairs in sync, run = flight phase or high speed,
                        walk_pace = everything else (lateral sequence, no flight).
                        Rules have no fitted parameters, so they are validated on the reference itself.
  kNN       (secondary) nearest reference windows in a standardized feature space
                        (joint angles, root velocity in heading frame, root height, foot heights).
                        Its nearest-neighbour distance also measures how far the motion is from
                        any reference motion: windows farther than 1.5 x the reference p95 are
                        labelled "off-manifold" (e.g. falling) and excluded from p(g).

Metrics per checkpoint:
  p(g)        fraction of on-manifold windows in each gait (footfall classifier)
  H_norm      entropy of p over the reference gaits only, / log(3)  (1 = equal use, 0 = one gait)
  p_stand     behaviour not in the reference (standing still or stepping in place, < 0.3 m/s)
  JS          Jensen-Shannon divergence to the reference gait distribution (base 2, in [0, 1])
  active      number of gaits with p(g) > 5%
  switch/min  gait changes per minute inside an episode (after a majority filter)
  off_manifold fraction of windows far from all reference motion
  knn_agree   agreement between footfall and kNN labels

Commanded speed (steering env only, where each env has its own random target speed):
  Only windows whose command was constant for the whole window plus --settle_sec before it are used,
  so transitions right after a target change do not count. Windows are grouped by commanded speed
  into the bins of --speed_bins.
  speed_bins.csv  per checkpoint and bin: p(g) over on-manifold windows, p_off, commanded speed,
                  velocity along the target direction, velocity error (commanded - achieved)
  MI_cmd_gait     mutual information I(speed bin; gait) in bits over on-manifold windows
                  (0 = the gait does not depend on the command, i.e. one gait for every speed)
  MI_norm         MI / H(gait): fraction of the gait uncertainty explained by the command
  v_abs_err       mean |commanded speed - velocity along the target direction|

Usage (from MimicKit/):
  python tools/gait_analysis/gait_collapse.py --run_dir output/go2_loc_amp_allgaits_s42
"""

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mimickit"))

import envs.base_env as base_env
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent
import util.mp_util as mp_util
import util.torch_util as torch_util
import util.util as util

# clip file name -> gait class.
# walk0..3 and pace have the same footfall pattern (lateral pairs in sync, no flight) and the same
# speed (~1.1 m/s) in this dataset, so they form one class. trot (diagonal pairs, ~1.9 m/s) and
# run (flight phase, ~3.2 m/s) are distinct.
GAIT_OF_CLIP = {
    "go2_walk0": "walk_pace", "go2_walk1": "walk_pace", "go2_walk2": "walk_pace", "go2_walk3": "walk_pace",
    "go2_pace": "walk_pace", "go2_trot": "trot", "go2_run": "run",
}
REF_GAITS = ["walk_pace", "trot", "run"]
# behaviour that is not in the reference: standing still (or stepping in place).
# A short flight phase at trot speed is a flying trot, not a separate gait: below RUN_SPEED_MIN the
# footfall pattern decides, whatever the flight fraction.
GAITS = REF_GAITS + ["stand"]

# foot order follows key_bodies in the env config: FR, FL, RR, RL
CONTACT_THR = 0.03          # m above the resting foot height
FLIGHT_MIN = 0.05           # fraction of the window with all four feet in the air
RUN_SPEED_MIN = 2.0         # m/s, a flight phase above this speed is running (reference run: ~3.2 m/s)
RUN_SPEED = 2.5             # m/s, always running above this speed
STAND_SPEED = 0.3           # m/s
TROT_DIAG_MINUS_LAT = 0.3   # diagonal-pair sync minus lateral-pair sync
OFF_MANIFOLD_FACTOR = 1.5   # x reference p95 nearest-neighbour distance


#####################################################################
# per-step motion features (shared by reference and policy)
#####################################################################

def step_features(root_pos, root_rot, root_vel, dof_pos, foot_pos):
    """[N, F]: dof positions, root velocity in heading frame (3), root height (1), foot heights (4)."""
    heading_inv = torch_util.calc_heading_quat_inv(root_rot)
    local_vel = torch_util.quat_rotate(heading_inv, root_vel)
    return torch.cat([dof_pos, local_vel, root_pos[..., 2:3], foot_pos[..., 2]], dim=-1)


#####################################################################
# reference windows from MoCap
#####################################################################

def build_reference(env, win_steps):
    motion_lib = env._motion_lib
    dt = env._engine.get_timestep()
    foot_ids = env._key_body_ids
    lengths = motion_lib.get_motion_lengths().cpu().numpy()
    weights = motion_lib.get_motion_weights().cpu().numpy()

    feats, labels, clip_ids, starts, clip_names = [], [], [], [], []
    for m in range(motion_lib.get_num_motions()):
        name = os.path.splitext(os.path.basename(motion_lib.get_motion_file(m)))[0]
        clip_names.append(name)
        loop = motion_lib.get_motion_loop_mode(torch.tensor([m], device=env._device)).item() == 1
        t_max = lengths[m] if loop else lengths[m] - (win_steps - 1) * dt
        t0s = np.arange(0.0, max(t_max, 1e-6), dt)

        t0 = torch.tensor(t0s, device=env._device, dtype=torch.float)
        times = t0[:, None] + dt * torch.arange(win_steps, device=env._device)[None, :]
        mids = torch.full_like(times, m, dtype=torch.long)
        root_pos, root_rot, root_vel, _, joint_rot, _ = motion_lib.calc_motion_frame(mids.flatten(), times.flatten())
        body_pos, _ = env._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        dof_pos = motion_lib.joint_rot_to_dof(joint_rot)
        f = step_features(root_pos, root_rot, root_vel, dof_pos, body_pos[:, foot_ids, :])

        feats.append(f.reshape(len(t0s), win_steps, -1))
        labels += [GAITS.index(GAIT_OF_CLIP[name])] * len(t0s)
        clip_ids += [m] * len(t0s)
        starts.append(t0s)

    gait_w = np.zeros(len(GAITS))
    for m, name in enumerate(clip_names):
        gait_w[GAITS.index(GAIT_OF_CLIP[name])] += weights[m]

    return {
        "feats": torch.cat(feats, dim=0),
        "labels": torch.tensor(labels, device=env._device),
        "clip_ids": torch.tensor(clip_ids, device=env._device),
        "starts": torch.tensor(np.concatenate(starts), device=env._device, dtype=torch.float),
        "clip_names": clip_names,
        "gait_dist": gait_w / gait_w.sum(),
    }


#####################################################################
# footfall classifier (primary)
#####################################################################

def footfall_stats(windows, nd, base):
    c = (windows[..., nd + 4:] - base) < CONTACT_THR                 # [N, W, 4] FR FL RR RL
    diag = 0.5 * ((c[..., 0] == c[..., 3]).float().mean(1) + (c[..., 1] == c[..., 2]).float().mean(1))
    lat = 0.5 * ((c[..., 0] == c[..., 2]).float().mean(1) + (c[..., 1] == c[..., 3]).float().mean(1))
    return {
        "duty": c.float().mean((1, 2)),
        "diag": diag,
        "lat": lat,
        "flight": (~c.any(-1)).float().mean(1),
        "speed": torch.linalg.norm(windows[..., nd:nd + 2], dim=-1).mean(1),
    }


def footfall_classify(st):
    pred = torch.full_like(st["duty"], GAITS.index("walk_pace"), dtype=torch.long)
    pred[(st["diag"] - st["lat"]) > TROT_DIAG_MINUS_LAT] = GAITS.index("trot")
    flight = st["flight"] > FLIGHT_MIN
    pred[(flight & (st["speed"] > RUN_SPEED_MIN)) | (st["speed"] > RUN_SPEED)] = GAITS.index("run")
    pred[st["speed"] < STAND_SPEED] = GAITS.index("stand")
    return pred


def resting_foot_height(windows, nd):
    """2nd percentile of foot height: the height of a foot on the ground for this source
    (MoCap feet sit at ~0.0 m, simulated feet at ~0.01-0.02 m because of the foot radius)."""
    return torch.quantile(windows[..., nd + 4:].flatten().float()[:1_000_000], 0.02)


#####################################################################
# kNN classifier (secondary) + distance to the reference manifold
#####################################################################

class KNNGait:
    def __init__(self, ref_feats, ref_labels, sub, k):
        self.sub, self.k = sub, k
        flat = ref_feats.reshape(-1, ref_feats.shape[-1])
        self.mean = flat.mean(0)
        self.std = flat.std(0).clamp_min(1e-3)
        self.ref = self._embed(ref_feats)
        self.labels = ref_labels

    def _embed(self, w):
        w = ((w - self.mean) / self.std)[:, ::self.sub, :]
        return w.reshape(w.shape[0], -1) / np.sqrt(w.shape[1])

    def query(self, windows, exclude_fn=None, batch=4096):
        preds, dists = [], []
        for i in range(0, windows.shape[0], batch):
            q = self._embed(windows[i:i + batch])
            d = torch.cdist(q, self.ref)
            if exclude_fn is not None:
                d = d.masked_fill(exclude_fn(i, i + q.shape[0]), float("inf"))
            dk, ik = torch.topk(d, self.k, dim=1, largest=False)
            votes = torch.nn.functional.one_hot(self.labels[ik], len(GAITS)).sum(1)
            preds.append(votes.argmax(1))
            dists.append(dk[:, 0])
        return torch.cat(preds), torch.cat(dists)


def confusion(true, pred):
    conf = np.zeros((len(GAITS), len(GAITS)), dtype=int)
    for t, p in zip(true, pred):
        conf[t, p] += 1
    return conf


def format_conf(title, conf):
    lines = [title, "            " + " ".join("{:>10}".format(g) for g in GAITS)]
    for i, g in enumerate(REF_GAITS):
        lines.append("{:>10}  ".format(g) + " ".join("{:>10d}".format(c) for c in conf[i]) +
                     "   recall={:.3f}".format(conf[i, i] / max(conf[i].sum(), 1)))
    return lines


#####################################################################
# policy rollout
#####################################################################

def has_speed_command(env):
    """True for envs with a per-env target speed (steering); location uses one fixed float."""
    s = getattr(env, "_tar_speed", None)
    return isinstance(s, torch.Tensor) and s.dim() == 1 and s.shape[0] == env.get_num_envs()


def rollout(env, agent, num_steps, win_steps, warmup_steps, stride_steps, settle_steps=0):
    char_id = env._get_char_id()
    foot_ids = env._key_body_ids
    n, dev, e = env.get_num_envs(), env._device, env._engine
    use_cmd = has_speed_command(env)

    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)
    obs, info = env.reset()

    hist = None
    since_reset = torch.zeros([n, num_steps], device=dev, dtype=torch.long)
    ep_id = torch.zeros([n, num_steps], device=dev, dtype=torch.long)
    curr_since = torch.zeros(n, device=dev, dtype=torch.long)
    curr_ep = torch.zeros(n, device=dev, dtype=torch.long)
    num_falls, num_eps = 0, 0

    if use_cmd:
        cmd_speed = torch.zeros([n, num_steps], device=dev)    # commanded speed in effect during step t
        v_along = torch.zeros([n, num_steps], device=dev)      # root velocity along the target direction
        since_cmd = torch.zeros([n, num_steps], device=dev, dtype=torch.long)
        curr_since_cmd = torch.zeros(n, device=dev, dtype=torch.long)
        prev_speed = torch.full([n], -1.0, device=dev)
        prev_dir = torch.zeros([n, 2], device=dev)

    with torch.no_grad():
        for t in range(num_steps):
            if use_cmd:
                # command seen by the policy for this step (it is resampled inside env.step or on reset;
                # a reset always resamples it, so a new episode also restarts the counter)
                speed_t = env._tar_speed.clone()
                dir_t = env._tar_dir.clone()
                changed = (speed_t != prev_speed) | (dir_t != prev_dir).any(-1)
                curr_since_cmd[changed] = 0
                prev_speed, prev_dir = speed_t, dir_t

            action, _ = agent._decide_action(obs, info)
            obs, _, done, info = env.step(action)

            if use_cmd:
                curr_since_cmd += 1
                cmd_speed[:, t] = speed_t
                since_cmd[:, t] = curr_since_cmd
                v_along[:, t] = (e.get_root_vel(char_id)[:, :2] * dir_t).sum(-1)

            f = step_features(e.get_root_pos(char_id), e.get_root_rot(char_id), e.get_root_vel(char_id),
                              e.get_dof_pos(char_id), e.get_body_pos(char_id)[:, foot_ids, :])
            if hist is None:
                hist = torch.zeros([n, num_steps, f.shape[-1]], device=dev)
            hist[:, t] = f
            curr_since += 1
            since_reset[:, t] = curr_since
            ep_id[:, t] = curr_ep

            done_ids = (done != base_env.DoneFlags.NULL.value).nonzero().flatten()
            if len(done_ids) > 0:
                num_falls += (done[done_ids] == base_env.DoneFlags.FAIL.value).sum().item()
                num_eps += len(done_ids)
                obs, info = env.reset(done_ids)
                curr_since[done_ids] = 0
                curr_ep[done_ids] += 1

    # windows ending at step t, fully inside one episode and after the warm-up
    ends = torch.arange(win_steps - 1, num_steps, stride_steps, device=dev)
    valid = since_reset[:, ends] >= max(warmup_steps, win_steps)
    env_idx, end_idx = valid.nonzero(as_tuple=True)
    end_t = ends[end_idx]
    offs = torch.arange(-win_steps + 1, 1, device=dev)
    windows = hist[env_idx[:, None], end_t[:, None] + offs[None, :]]

    out = {"windows": windows, "env": env_idx, "end": end_t, "ep": ep_id[env_idx, end_t],
           "falls": num_falls, "episodes": num_eps + n}
    if use_cmd:
        idx = (env_idx[:, None], end_t[:, None] + offs[None, :])
        out["cmd"] = cmd_speed[idx].mean(1)
        out["v_along"] = v_along[idx].mean(1)
        # command constant over the whole window and for settle_steps before it
        out["cmd_ok"] = since_cmd[env_idx, end_t] >= win_steps + settle_steps
    return out


#####################################################################
# metrics
#####################################################################

def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def js_div(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def switches_per_min(pred, env_idx, ep, end, stride_steps, dt, smooth=5):
    """Gait changes per minute inside episodes, after a sliding majority filter of `smooth` windows."""
    key = env_idx.astype(np.int64) * 100000 + ep
    order = np.lexsort((end, key))
    pred, key = pred[order], key[order]
    total_sw, total_time = 0, 0.0
    for k in np.unique(key):
        seq = pred[key == k]
        if len(seq) < smooth:
            continue
        h = smooth // 2
        sm = np.array([np.bincount(seq[max(0, i - h): i + h + 1], minlength=len(GAITS)).argmax()
                       for i in range(len(seq))])
        total_sw += int((sm[1:] != sm[:-1]).sum())
        total_time += len(seq) * stride_steps * dt
    return 60.0 * total_sw / max(total_time, 1e-6)


def mutual_info(a, b, na, nb):
    """Plug-in mutual information I(a; b) in bits and the entropy H(b) in bits, for integer labels."""
    if len(a) == 0:
        return float("nan"), float("nan")
    joint = np.zeros((na, nb))
    np.add.at(joint, (a, b), 1.0)
    joint /= joint.sum()
    pa, pb = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = float((joint[nz] * np.log2(joint[nz] / (pa[:, None] * pb[None, :])[nz])).sum())
    hb = float(-(pb[pb > 0] * np.log2(pb[pb > 0])).sum())
    return mi, hb


def speed_bin_rows(it, edges, cmd, v_along, ff, on):
    """One row per commanded-speed bin: p_* over on-manifold windows, p_off over all windows of the bin."""
    nan = float("nan")
    rows = []
    b = np.clip(np.digitize(cmd, edges) - 1, 0, len(edges) - 2)
    for i in range(len(edges) - 1):
        m = b == i
        mo = m & on
        p = np.bincount(ff[mo], minlength=len(GAITS)) / mo.sum() if mo.any() else np.full(len(GAITS), nan)
        rows.append({
            "iter": it, "bin": i, "v_lo": edges[i], "v_hi": edges[i + 1], "windows": int(m.sum()),
            **{"p_" + g: p[k] for k, g in enumerate(GAITS)},
            "p_off": 1.0 - on[m].mean() if m.any() else nan,
            "v_cmd": float(cmd[m].mean()) if m.any() else nan,
            "v_along": float(v_along[m].mean()) if m.any() else nan,
            "v_err": float((cmd[m] - v_along[m]).mean()) if m.any() else nan,
            "v_abs_err": float(np.abs(cmd[m] - v_along[m]).mean()) if m.any() else nan,
        })
    return rows


def iter_of(path):
    return int(re.findall(r"(\d+)", os.path.basename(path))[-1])


def samples_by_iter(run_dir):
    """Iteration -> Samples from the training log.txt (rows are separated by \\r)."""
    path = os.path.join(run_dir, "log.txt")
    if not os.path.exists(path):
        return {}
    with open(path, "r", newline="") as f:
        rows = [r.split() for r in re.split(r"[\r\n]+", f.read()) if r.strip()]
    head = rows[0]
    i_it, i_s = head.index("Iteration"), head.index("Samples")
    return {int(r[i_it]): int(float(r[i_s])) for r in rows[1:] if len(r) == len(head)}


#####################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--rollout_sec", type=float, default=20.0)
    ap.add_argument("--warmup_sec", type=float, default=2.0)
    ap.add_argument("--win_sec", type=float, default=1.0)
    ap.add_argument("--stride_steps", type=int, default=3)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--sub", type=int, default=2, help="temporal subsampling inside a window for the kNN")
    ap.add_argument("--every", type=int, default=1, help="analyse every n-th checkpoint (the last is always included)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed_bins", default="0.6,1.3,2.0,2.6,3.2",
                    help="commanded-speed bin edges in m/s (steering only); reference: walk/pace ~1.1, trot ~1.9, run ~3.2")
    ap.add_argument("--settle_sec", type=float, default=1.0,
                    help="ignore windows that start less than this after a target change (steering only)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, "gait_analysis")
    os.makedirs(out_dir, exist_ok=True)

    mp_util.init(0, 1, args.device, None)
    util.set_rand_seed(args.seed)

    env = env_builder.build_env(os.path.join(args.run_dir, "env_config.yaml"),
                                os.path.join(args.run_dir, "engine_config.yaml"),
                                args.num_envs, args.device, visualize=False)
    agent = agent_builder.build_agent(os.path.join(args.run_dir, "agent_config.yaml"), env, args.device)

    dt = env._engine.get_timestep()
    nd = env._engine.get_dof_pos(env._get_char_id()).shape[-1]
    win_steps = int(round(args.win_sec / dt))
    steps = int(round(args.rollout_sec / dt))
    warmup = int(round(args.warmup_sec / dt))
    settle = int(round(args.settle_sec / dt))
    edges = np.array([float(v) for v in args.speed_bins.split(",")])
    use_cmd = has_speed_command(env)
    if use_cmd:
        print("Steering env: recording the commanded speed, bins (m/s):", edges)
    log_samples = samples_by_iter(args.run_dir)

    # ---------------- reference + validation
    ref = build_reference(env, win_steps)
    ref_base = resting_foot_height(ref["feats"], nd)
    ref_st = footfall_stats(ref["feats"], nd, ref_base)
    ref_ff = footfall_classify(ref_st).cpu().numpy()
    ref_true = ref["labels"].cpu().numpy()

    # Reference gait distribution measured with the SAME footfall classifier as the policy:
    # each clip contributes its dataset weight, split by the gaits found inside it (some "walk"
    # clips contain trotting segments). Clip-label based distribution kept for the report.
    clip_np = ref["clip_ids"].cpu().numpy()
    clip_w = env._motion_lib.get_motion_weights().cpu().numpy()
    q = np.zeros(len(GAITS))
    for m in range(len(ref["clip_names"])):
        q += clip_w[m] * np.bincount(ref_ff[clip_np == m], minlength=len(GAITS)) / (clip_np == m).sum()
    ref["gait_dist_labels"] = ref["gait_dist"]
    ref["gait_dist"] = q / q.sum()

    clf = KNNGait(ref["feats"], ref["labels"], args.sub, args.knn)
    clip, start = ref["clip_ids"], ref["starts"]
    excl = lambda a, b: (clip[a:b, None] == clip[None, :]) & ((start[a:b, None] - start[None, :]).abs() < args.win_sec)
    ref_knn, ref_nn = clf.query(ref["feats"], excl)
    off_thr = OFF_MANIFOLD_FACTOR * torch.quantile(ref_nn[torch.isfinite(ref_nn)], 0.95).item()

    lines = ["Reference: {} windows of {:.1f} s".format(len(ref_true), args.win_sec),
             "Reference gait distribution, clip labels x dataset weights: " +
             ", ".join("{}={:.3f}".format(g, p) for g, p in zip(GAITS, ref["gait_dist_labels"])),
             "Reference gait distribution, footfall classifier x dataset weights (used for JS): " +
             ", ".join("{}={:.3f}".format(g, p) for g, p in zip(GAITS, ref["gait_dist"])), ""]
    lines.append("Footfall statistics of the reference (mean per gait):")
    for i, g in enumerate(REF_GAITS):
        m = ref_true == i
        lines.append("  {:>10}: n={:4d} duty={:.2f} diag={:.2f} lat={:.2f} flight={:.2f} speed={:.2f} m/s".format(
            g, m.sum(), *[ref_st[k][torch.from_numpy(m).to(clip.device)].mean().item()
                          for k in ["duty", "diag", "lat", "flight", "speed"]]))
    lines.append("")
    lines += format_conf("Footfall classifier on the reference (rows=true, cols=pred):", confusion(ref_true, ref_ff))
    lines.append("")
    lines += format_conf("kNN on the reference, overlap-excluded (rows=true, cols=pred):",
                         confusion(ref_true, ref_knn.cpu().numpy()))
    lines.append("  (trot and run are single short clips: once overlapping windows are excluded, few or no")
    lines.append("   same-class neighbours remain, so kNN recall for them is a pessimistic lower bound)")
    lines.append("")
    lines.append("Off-manifold threshold (kNN distance): {:.2f}  (= {} x reference p95)".format(off_thr, OFF_MANIFOLD_FACTOR))
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, "validation.txt"), "w") as f:
        f.write(report + "\n")

    # ---------------- checkpoints
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "int_models", "model_*.pt")), key=iter_of)
    sel = ckpts[::args.every]
    if ckpts[-1] not in sel:
        sel.append(ckpts[-1])

    rows, bin_rows, per_ckpt = [], [], {}
    for ck in sel:
        it = iter_of(ck)
        util.set_rand_seed(args.seed)
        agent.load(ck)
        r = rollout(env, agent, steps, win_steps, warmup, args.stride_steps, settle)
        if r["windows"].shape[0] == 0:
            print("iter {}: no valid windows".format(it))
            continue

        st = footfall_stats(r["windows"], nd, resting_foot_height(r["windows"], nd))
        ff = footfall_classify(st).cpu().numpy()
        knn, nn = clf.query(r["windows"])
        knn, nn = knn.cpu().numpy(), nn.cpu().numpy()
        on = nn <= off_thr

        nan = float("nan")
        ref_on = on & (ff < len(REF_GAITS))   # windows in a reference gait (kNN only knows these)
        p = np.bincount(ff[on], minlength=len(GAITS)) / on.sum() if on.any() else np.full(len(GAITS), nan)
        row = {
            "iter": it,
            "samples": log_samples.get(it, -1),
            **{"p_" + g: p[i] for i, g in enumerate(GAITS)},
            "H_norm": entropy(p[:len(REF_GAITS)] / p[:len(REF_GAITS)].sum()) / np.log(len(REF_GAITS)) if p[:len(REF_GAITS)].sum() > 0 else nan,
            "JS": js_div(p, ref["gait_dist"]) if on.any() else nan,
            "active": int((p[:len(REF_GAITS)] > 0.05).sum()),
            "switch_per_min": switches_per_min(ff[on], r["env"].cpu().numpy()[on], r["ep"].cpu().numpy()[on],
                                               r["end"].cpu().numpy()[on], args.stride_steps, dt),
            "off_manifold": 1.0 - on.mean(),
            "knn_agree": float((knn[ref_on] == ff[ref_on]).mean()) if ref_on.any() else nan,
            "fall_rate": r["falls"] / r["episodes"],
            "speed_median": float(np.median(st["speed"].cpu().numpy()[on])) if on.any() else float("nan"),
            "windows": len(ff),
        }
        extra = {}
        if use_cmd:
            cmd, va, ok = r["cmd"].cpu().numpy(), r["v_along"].cpu().numpy(), r["cmd_ok"].cpu().numpy()
            b = np.clip(np.digitize(cmd, edges) - 1, 0, len(edges) - 2)
            sel_mi = ok & on
            mi, hg = mutual_info(b[sel_mi], ff[sel_mi], len(edges) - 1, len(GAITS))
            row["MI_cmd_gait"] = mi
            row["MI_norm"] = mi / hg if hg > 0 else float("nan")
            row["v_abs_err"] = float(np.abs(cmd[ok] - va[ok]).mean()) if ok.any() else float("nan")
            bin_rows += speed_bin_rows(it, edges, cmd[ok], va[ok], ff[ok], on[ok])
            extra = {"cmd": cmd, "v_along": va, "cmd_ok": ok}
        rows.append(row)
        print("iter {:5d} | ".format(it) + " ".join("{}={:.2f}".format(g, p[i]) for i, g in enumerate(GAITS)) +
              " | H={:.3f} JS={:.3f} active={} sw/min={:.1f} off={:.2f} agree={:.2f} falls={:.2f} v={:.2f}".format(
                  row["H_norm"], row["JS"], row["active"], row["switch_per_min"], row["off_manifold"],
                  row["knn_agree"], row["fall_rate"], row["speed_median"]) +
              (" | MI={:.3f} bits ({:.0%} of H) |v err|={:.2f}".format(row["MI_cmd_gait"], row["MI_norm"], row["v_abs_err"])
               if use_cmd else ""))
        per_ckpt[it] = {"ff": ff, "on": on, **{k: v.cpu().numpy() for k, v in st.items()}, **extra}

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if bin_rows:
        with open(os.path.join(out_dir, "speed_bins.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
            w.writeheader()
            w.writerows(bin_rows)

    last = max(per_ckpt)
    np.savez(os.path.join(out_dir, "windows.npz"), gaits=np.array(GAITS), ref_gait_dist=ref["gait_dist"],
             ref_labels=ref_true, **{"ref_" + k: v.cpu().numpy() for k, v in ref_st.items()},
             final_iter=last, speed_bins=edges if use_cmd else np.zeros(0),
             **{"final_" + k: v for k, v in per_ckpt[last].items()})
    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
