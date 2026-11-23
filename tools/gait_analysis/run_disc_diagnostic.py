"""
Diagnostic 2: what the discriminator thinks of each reference clip, and of the policy's own motion.

Question: does the discriminator separate the run clip easily, i.e. does it "know" that the agent
never produces running?

For each checkpoint in --iters:
  demo     every window of every MoCap clip (start times on a grid of one control step over the whole
           clip, the same windows the AMP samples for its demo batch) goes through the discriminator
           of that checkpoint, with the checkpoint's own disc-observation normalizer
  policy   the policy is rolled out in the steering env (random commands, as in training); the
           discriminator logit of each step is kept and grouped by the gait of the 1-s window that
           ends at that step (footfall classifier of gait_collapse.py)

Reported per clip and per policy gait: logit (mean, quartiles), D = sigmoid(logit) and the style
reward the AMP gives for that transition, r = -log(max(1 - D, 1e-4)) * disc_reward_scale, exactly
as in AMPAgent._calc_disc_rewards.

Expected signature of mode collapse (mode seeking): walk logits near 0 (the agent covers walking,
the discriminator cannot tell it apart) and run logits clearly positive (sure it is real, since the
agent never produces it), with the run logit growing over training.

Nothing in MimicKit is modified; the script only calls the agent's model and normalizer.

Outputs in <run_dir>/mode_diagnostics/:
  disc_logits.csv  per checkpoint and source (clip or policy gait): n, logit mean / p25 / p50 / p75,
                   mean D, mean style reward
  disc_logits.txt  summary table
  disc_logits.png  (a) logit distribution per clip and per policy gait, final checkpoint;
                   (b) mean logit per gait over training

Usage (from MimicKit/):
  python tools/gait_analysis/run_disc_diagnostic.py --run_dir output/go2_steer_amp_allgaits_2048env_s42
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "mimickit"))

import gait_collapse as gc  # noqa: E402
import envs.base_env as base_env  # noqa: E402
import envs.env_builder as env_builder  # noqa: E402
import learning.agent_builder as agent_builder  # noqa: E402
import learning.base_agent as base_agent  # noqa: E402
import util.mp_util as mp_util  # noqa: E402
import util.torch_util as torch_util  # noqa: E402
import util.util as util  # noqa: E402


def disc_logits(agent, disc_obs):
    with torch.no_grad():
        norm = agent._disc_obs_norm.normalize(disc_obs)
        out = torch_util.eval_minibatch(agent._model.eval_disc, {"disc_obs": norm}, agent._disc_eval_batch_size)
    return out.squeeze(-1)


def style_reward(agent, logits):
    prob = torch.sigmoid(logits)
    return -torch.log(torch.clamp_min(1 - prob, 1e-4)) * agent._disc_reward_scale


def demo_windows(env):
    """Disc observations of every window of every clip: {clip name: tensor [N, obs]}."""
    ml, dt = env._motion_lib, env._engine.get_timestep()
    out = {}
    for m in range(ml.get_num_motions()):
        name = os.path.splitext(os.path.basename(ml.get_motion_file(m)))[0]
        length = ml.get_motion_lengths()[m].item()
        t0 = torch.arange(0.0, length, dt, device=env._device)
        ids = torch.full_like(t0, m, dtype=torch.long)
        out[name] = env._compute_disc_obs_demo(ids, t0)
    return out


def policy_rollout(env, agent, num_steps, warmup_steps, win_steps, stride_steps):
    """Logit of every step + gait label of the 1-s window ending there (same rules as gait_collapse.rollout)."""
    char_id, foot_ids = env._get_char_id(), env._key_body_ids
    n, dev, e = env.get_num_envs(), env._device, env._engine
    nd = e.get_dof_pos(char_id).shape[-1]

    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)
    obs, info = env.reset()
    hist, logit = None, torch.zeros([n, num_steps], device=dev)
    since = torch.zeros([n, num_steps], device=dev, dtype=torch.long)
    curr = torch.zeros(n, device=dev, dtype=torch.long)
    with torch.no_grad():
        for t in range(num_steps):
            action, _ = agent._decide_action(obs, info)
            obs, _, done, info = env.step(action)
            f = gc.step_features(e.get_root_pos(char_id), e.get_root_rot(char_id), e.get_root_vel(char_id),
                                 e.get_dof_pos(char_id), e.get_body_pos(char_id)[:, foot_ids, :])
            if hist is None:
                hist = torch.zeros([n, num_steps, f.shape[-1]], device=dev)
            hist[:, t] = f
            # disc obs of the transition that just happened (before any reset overwrites it)
            logit[:, t] = disc_logits(agent, info["disc_obs"])
            curr += 1
            since[:, t] = curr
            done_ids = (done != base_env.DoneFlags.NULL.value).nonzero().flatten()
            if len(done_ids) > 0:
                obs, info = env.reset(done_ids)
                curr[done_ids] = 0

    ends = torch.arange(win_steps - 1, num_steps, stride_steps, device=dev)
    valid = since[:, ends] >= max(warmup_steps, win_steps)
    env_idx, end_idx = valid.nonzero(as_tuple=True)
    end_t = ends[end_idx]
    offs = torch.arange(-win_steps + 1, 1, device=dev)
    windows = hist[env_idx[:, None], end_t[:, None] + offs[None, :]]
    st = gc.footfall_stats(windows, nd, gc.resting_foot_height(windows, nd))
    return gc.footfall_classify(st), logit[env_idx, end_t]


def summarize(agent, logits):
    lg = logits.float()
    q = torch.quantile(lg, torch.tensor([0.25, 0.5, 0.75], device=lg.device)).tolist() if len(lg) else [np.nan] * 3
    return {"n": int(len(lg)),
            "logit_mean": lg.mean().item() if len(lg) else np.nan,
            "logit_p25": q[0], "logit_p50": q[1], "logit_p75": q[2],
            "D_mean": torch.sigmoid(lg).mean().item() if len(lg) else np.nan,
            "reward_mean": style_reward(agent, lg).mean().item() if len(lg) else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--iters", default="500,1000,1500,2288", help="checkpoint iterations (int_models/model_<iter>.pt)")
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--rollout_sec", type=float, default=10.0)
    ap.add_argument("--warmup_sec", type=float, default=2.0)
    ap.add_argument("--win_sec", type=float, default=1.0)
    ap.add_argument("--stride_steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = os.path.join(args.run_dir, "mode_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    mp_util.init(0, 1, args.device, None)
    util.set_rand_seed(args.seed)
    env = env_builder.build_env(os.path.join(args.run_dir, "env_config.yaml"),
                                os.path.join(args.run_dir, "engine_config.yaml"),
                                args.num_envs, args.device, visualize=False)
    agent = agent_builder.build_agent(os.path.join(args.run_dir, "agent_config.yaml"), env, args.device)

    dt = env._engine.get_timestep()
    steps, warmup, win = (int(round(s / dt)) for s in (args.rollout_sec, args.warmup_sec, args.win_sec))
    demo = demo_windows(env)

    rows, per_iter = [], {}
    for it in [int(s) for s in args.iters.split(",")]:
        ck = os.path.join(args.run_dir, "int_models", "model_{:010d}.pt".format(it))
        util.set_rand_seed(args.seed)
        agent.load(ck)
        store = {}
        for name, obs in demo.items():
            lg = disc_logits(agent, obs)
            store["demo:" + name] = lg.cpu().numpy()
            rows.append({"iter": it, "source": "demo", "name": name, "gait": gc.GAIT_OF_CLIP[name], **summarize(agent, lg)})
        lab, lg = policy_rollout(env, agent, steps, warmup, win, args.stride_steps)
        for i, g in enumerate(gc.GAITS):
            m = lab == i
            store["policy:" + g] = lg[m].cpu().numpy()
            rows.append({"iter": it, "source": "policy", "name": "policy " + g, "gait": g, **summarize(agent, lg[m])})
        per_iter[it] = store
        print("iter {} done".format(it))

    with open(os.path.join(out_dir, "disc_logits.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["Diagnostic 2: discriminator logit per reference clip and per policy gait",
             "D = sigmoid(logit) = probability the discriminator assigns to 'real MoCap'; "
             "style reward = -log(max(1-D,1e-4)) x {}".format(agent._disc_reward_scale), ""]
    for it in per_iter:
        lines.append("Checkpoint iter {}".format(it))
        lines.append("  {:<22} {:>6} {:>8} {:>8} {:>8} {:>8} {:>7} {:>7}".format(
            "source", "n", "mean", "p25", "p50", "p75", "D", "reward"))
        for r in rows:
            if r["iter"] != it or r["n"] == 0:
                continue
            lines.append("  {:<22} {:>6d} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>7.2f} {:>7.2f}".format(
                ("MoCap " if r["source"] == "demo" else "") + r["name"], r["n"], r["logit_mean"], r["logit_p25"],
                r["logit_p50"], r["logit_p75"], r["D_mean"], r["reward_mean"]))
        lines.append("")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, "disc_logits.txt"), "w") as f:
        f.write(report + "\n")

    # ---------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_gait_collapse as pg

    last = max(per_iter)
    store = per_iter[last]
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 3.8), gridspec_kw={"width_ratios": [1.5, 1.0]})
    keys = [k for k in store if k.startswith("demo:")] + \
           ["policy:" + g for g in gc.GAITS if len(store["policy:" + g]) > 0]
    data = [store[k] for k in keys]
    bp = axs[0].boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, k in zip(bp["boxes"], keys):
        g = gc.GAIT_OF_CLIP[k[5:]] if k.startswith("demo:") else k[7:]
        patch.set_facecolor(pg.GAIT_COLORS[g])
        patch.set_alpha(0.85 if k.startswith("demo:") else 0.35)
        patch.set_edgecolor(pg.INK2)
    for med in bp["medians"]:
        med.set_color(pg.INK)
    labels = [k[5:].replace("go2_", "MoCap\n") if k.startswith("demo:") else "policy\n" + k[7:] for k in keys]
    axs[0].set_xticks(range(1, len(keys) + 1))
    axs[0].set_xticklabels(labels, fontsize=8)
    axs[0].axhline(0, color=pg.INK2, lw=0.8, ls="--")
    axs[0].text(len(keys) + 0.4, 0, "D = 0.5", color=pg.INK2, fontsize=8, ha="right", va="bottom")
    axs[0].set_ylabel("discriminator logit")
    axs[0].set_title("Discriminator logit per clip and policy gait (iter {})".format(last), loc="left", color=pg.INK)
    axs[0].grid(axis="x", visible=False)

    its = sorted(per_iter)
    for g in gc.REF_GAITS:
        ys = [np.mean(np.concatenate([per_iter[i]["demo:" + n] for n, gg in gc.GAIT_OF_CLIP.items() if gg == g
                                      and "demo:" + n in per_iter[i]])) for i in its]
        axs[1].plot(its, ys, color=pg.GAIT_COLORS[g], lw=2.0, marker="o", label="MoCap " + pg.LABELS[g])
    pol = [np.mean(np.concatenate([v for k, v in per_iter[i].items() if k.startswith("policy:")])) for i in its]
    axs[1].plot(its, pol, color=pg.INK2, lw=1.5, ls="--", marker="o", label="policy (all gaits)")
    axs[1].axhline(0, color=pg.INK2, lw=0.8, ls=":")
    axs[1].set_xlabel("training iteration")
    axs[1].set_ylabel("mean discriminator logit")
    axs[1].set_title("Mean logit over training", loc="left", color=pg.INK)
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "disc_logits.png"), bbox_inches="tight")
    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
