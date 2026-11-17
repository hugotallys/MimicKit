"""
Example videos of each gait produced by a policy checkpoint.

1. Roll out the checkpoint (deterministic actions) and store the full state of every robot.
2. Label every 1-second window with the footfall classifier of gait_collapse.py and smooth the
   labels over time.
3. For each gait, pick the longest stretches of one robot staying in that gait.
4. Replay each stretch kinematically on robot 0 (pose by pose, no physics, so it is exact) and
   render it with a camera that follows the robot from its side.

Usage (from MimicKit/):
  python tools/gait_analysis/gait_videos.py --run_dir output/go2_loc_amp_allgaits_s42
  python tools/gait_analysis/gait_videos.py --run_dir output/go2_loc_amp_allgaits_s42 --iter 1200
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gait_collapse as gc  # noqa: E402  (also puts mimickit/ on sys.path)

import envs.base_env as base_env  # noqa: E402
import envs.env_builder as env_builder  # noqa: E402
import learning.agent_builder as agent_builder  # noqa: E402
import learning.base_agent as base_agent  # noqa: E402
import util.mp_util as mp_util  # noqa: E402
import util.torch_util as torch_util  # noqa: E402
import util.util as util  # noqa: E402
import util.video as video  # noqa: E402

LABEL_TEXT = {"walk_pace": "walk / pace", "trot": "trot", "run": "run",
              "stand": "stand (not in reference)", "hop": "hop (not in reference)"}


def rollout_states(env, agent, num_steps):
    """Roll out and keep root pos/rot, dof pos and step features of every robot at every step."""
    char_id = env._get_char_id()
    foot_ids = env._key_body_ids
    n, dev, e = env.get_num_envs(), env._device, env._engine

    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)
    obs, info = env.reset()

    root_pos, root_rot, dof_pos, feats = [], [], [], []
    since = torch.zeros([n, num_steps], device=dev, dtype=torch.long)
    curr = torch.zeros(n, device=dev, dtype=torch.long)
    with torch.no_grad():
        for t in range(num_steps):
            action, _ = agent._decide_action(obs, info)
            obs, _, done, info = env.step(action)

            rp, rr, dp = e.get_root_pos(char_id).clone(), e.get_root_rot(char_id).clone(), e.get_dof_pos(char_id).clone()
            root_pos.append(rp)
            root_rot.append(rr)
            dof_pos.append(dp)
            feats.append(gc.step_features(rp, rr, e.get_root_vel(char_id), dp, e.get_body_pos(char_id)[:, foot_ids, :]))
            curr += 1
            since[:, t] = curr

            done_ids = (done != base_env.DoneFlags.NULL.value).nonzero().flatten()
            if len(done_ids) > 0:
                obs, info = env.reset(done_ids)
                curr[done_ids] = 0

    return {"root_pos": torch.stack(root_pos, 1), "root_rot": torch.stack(root_rot, 1),
            "dof_pos": torch.stack(dof_pos, 1), "feats": torch.stack(feats, 1), "since": since}


def label_all(st, nd, win, warmup):
    """Gait label of the window ending at each step (-1 where the window is not valid)."""
    feats, since = st["feats"], st["since"]
    n, T, _ = feats.shape
    labels = torch.full([n, T], -1, dtype=torch.long, device=feats.device)
    speed = torch.zeros([n, T], device=feats.device)
    ends = torch.arange(win - 1, T, device=feats.device)
    offs = torch.arange(-win + 1, 1, device=feats.device)
    windows = feats[:, ends[:, None] + offs[None, :]]           # [n, E, win, F]
    flat = windows.reshape(-1, win, feats.shape[-1])
    base = gc.resting_foot_height(flat, nd)
    s = gc.footfall_stats(flat, nd, base)
    lab = gc.footfall_classify(s).reshape(n, -1)
    valid = since[:, ends] >= max(win, warmup)
    labels[:, ends] = torch.where(valid, lab, torch.full_like(lab, -1))
    speed[:, ends] = s["speed"].reshape(n, -1)
    return labels.cpu().numpy(), speed.cpu().numpy()


def smooth(seq, k):
    out = seq.copy()
    h = k // 2
    for i in range(len(seq)):
        w = seq[max(0, i - h): i + h + 1]
        w = w[w >= 0]
        out[i] = np.bincount(w).argmax() if len(w) else -1
    return out


def find_runs(labels, cls, min_len):
    """All (env, start, end) stretches where the smoothed label stays == cls, longest first."""
    runs = []
    for e in range(labels.shape[0]):
        seq = labels[e] == cls
        t = 0
        while t < len(seq):
            if seq[t]:
                s = t
                while t < len(seq) and seq[t]:
                    t += 1
                if t - s >= min_len:
                    runs.append((t - s, e, s, t))
            else:
                t += 1
    runs.sort(reverse=True)
    return runs


def render_clip(env, st, e, t0, t1, text_fn, cam_offset_local):
    eng = env._engine
    rec = eng._video_recorder
    char_id = env._get_char_id()
    env0 = torch.tensor([0], device=env._device)

    # hide every other robot far below the ground so only the replayed one is in view
    others = torch.arange(1, env.get_num_envs(), device=env._device)
    if len(others) > 0:
        hidden = torch.zeros([len(others), 3], device=env._device)
        hidden[:, 2] = -50.0
        eng.set_root_pos(others, char_id, hidden)

    frames = []
    for t in range(t0, t1):
        rp, rr, dp = st["root_pos"][e, t], st["root_rot"][e, t], st["dof_pos"][e, t]
        eng.set_root_pos(env0, char_id, rp[None])
        eng.set_root_rot(env0, char_id, rr[None])
        eng.set_dof_pos(env0, char_id, dp[None])
        eng._sim_state.eval_fk()

        # camera on the robot's side: rotate the local offset by the robot's heading
        heading = torch_util.calc_heading(rr[None]).item()
        c, s = np.cos(heading), np.sin(heading)
        off = np.array([c * cam_offset_local[0] - s * cam_offset_local[1],
                        s * cam_offset_local[0] + c * cam_offset_local[1],
                        cam_offset_local[2]])
        rec._cam_pos = rec._cam_target + off

        img = Image.fromarray(rec._record_frame())
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width, 34], fill=(20, 20, 20))
        draw.text((10, 8), text_fn(t), fill=(255, 255, 255), font=FONT)
        frames.append(np.asarray(img))
    return frames


def load_font():
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, 18)
        except OSError:
            pass
    return ImageFont.load_default()


FONT = load_font()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--iter", type=int, default=None, help="checkpoint iteration (default: last)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--rollout_sec", type=float, default=20.0)
    ap.add_argument("--warmup_sec", type=float, default=2.0)
    ap.add_argument("--win_sec", type=float, default=1.0)
    ap.add_argument("--smooth_sec", type=float, default=0.5)
    ap.add_argument("--min_sec", type=float, default=1.5, help="shortest stretch worth a video")
    ap.add_argument("--max_sec", type=float, default=6.0, help="longest video")
    ap.add_argument("--per_gait", type=int, default=2, help="videos per gait (from different robots)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "int_models", "model_*.pt")), key=gc.iter_of)
    ckpt = ckpts[-1] if args.iter is None else next(c for c in ckpts if gc.iter_of(c) == args.iter)
    it = gc.iter_of(ckpt)
    out_dir = args.out_dir or os.path.join("output", "figures", "gait_videos_" + os.path.basename(args.run_dir.rstrip("/\\")),
                                           "iter{:05d}".format(it))
    os.makedirs(out_dir, exist_ok=True)

    mp_util.init(0, 1, args.device, None)
    util.set_rand_seed(args.seed)
    env = env_builder.build_env(os.path.join(args.run_dir, "env_config.yaml"),
                                os.path.join(args.run_dir, "engine_config.yaml"),
                                args.num_envs, args.device, visualize=False, record_video=True)
    agent = agent_builder.build_agent(os.path.join(args.run_dir, "agent_config.yaml"), env, args.device)
    agent.load(ckpt)

    rec = env._engine._video_recorder
    rec._cam_target = np.array([0.0, 0.0, 0.25])
    cam_offset_local = np.array([0.3, -1.6, 0.35])   # slightly ahead, to the robot's right, a bit above

    dt = env._engine.get_timestep()
    nd = env._engine.get_dof_pos(env._get_char_id()).shape[-1]
    win = int(round(args.win_sec / dt))
    st = rollout_states(env, agent, int(round(args.rollout_sec / dt)))
    labels, speed = label_all(st, nd, win, int(round(args.warmup_sec / dt)))
    k = max(1, int(round(args.smooth_sec / dt)))
    sm = np.stack([smooth(l, k) for l in labels])

    counts = np.bincount(sm[sm >= 0], minlength=len(gc.GAITS))
    print("iter {}: windows per gait (smoothed): ".format(it) +
          ", ".join("{}={}".format(g, c) for g, c in zip(gc.GAITS, counts)))

    for ci, g in enumerate(gc.GAITS):
        runs = find_runs(sm, ci, int(round(args.min_sec / dt)))
        used_envs, n_made = set(), 0
        for length, e, s, t in runs:
            if e in used_envs or n_made >= args.per_gait:
                continue
            used_envs.add(e)
            # a label at step t describes the window [t - win + 1, t]: show the motion those windows cover
            f0 = max(0, s - win // 2)
            f1 = min(sm.shape[1], f0 + int(round(args.max_sec / dt)), t)
            text = lambda tt, e=e: "{}  |  iter {}  |  robot {}  |  {:.2f} m/s".format(
                LABEL_TEXT[g], it, e, speed[e, max(tt, win - 1)])
            frames = render_clip(env, st, e, f0, f1, text, cam_offset_local)
            vid = video.Video(int(round(1.0 / dt)))
            for fr in frames:
                vid.add_frame(fr)
            path = os.path.join(out_dir, "{}_{}_robot{:03d}.mp4".format(ci, g, e))
            vid.save(path)
            n_made += 1
            print("  {:>10}: robot {:3d}, {:.1f} s -> {}".format(g, e, (f1 - f0) * dt, path))
        if n_made == 0:
            print("  {:>10}: no stretch of at least {:.1f} s".format(g, args.min_sec))


if __name__ == "__main__":
    main()
