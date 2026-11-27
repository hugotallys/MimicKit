"""
Per-frame AMP discriminator features for the motion VQ-VAE, for MoCap windows and for policy rollouts.

A window of W frames is expressed exactly as the AMP discriminator expresses its own window: every frame
relative to the root position and heading of the window's last frame (envs/amp_env.compute_disc_obs).
The discriminator observation of a W-step window is [W x C] flattened, so it is reshaped to (W, C).

MoCap windows come from env._compute_disc_obs_demo with num_disc_obs_steps temporarily set to W.
Rollout windows are assembled from the newest entry of the env's discriminator history buffers, recorded
at every step, and passed through the same compute_disc_obs. MimicKit itself is not modified.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gait_analysis"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "mimickit"))

import envs.amp_env as amp_env  # noqa: E402
import envs.env_builder as env_builder  # noqa: E402


def build_env(run_dir, num_envs, device):
    return env_builder.build_env(os.path.join(run_dir, "env_config.yaml"), os.path.join(run_dir, "engine_config.yaml"),
                                 num_envs, device, visualize=False)


def mocap_windows(env, clip_ids, t_end, W, chunk=4096):
    """(N, W, C) AMP features of the W-frame MoCap windows ending at t_end."""
    old = env._num_disc_obs_steps
    env._num_disc_obs_steps = W
    out = []
    try:
        for i in range(0, len(t_end), chunk):
            out.append(env._compute_disc_obs_demo(clip_ids[i:i + chunk], t_end[i:i + chunk]))
    finally:
        env._num_disc_obs_steps = old
    x = torch.cat(out)
    return x.view(x.shape[0], W, -1)


def sample_mocap(env, n, W, gen=None):
    """Random MoCap windows, clips drawn with the dataset weights (as the AMP demo sampler)."""
    lib = env._motion_lib
    dt = env._engine.get_timestep()
    w = lib.get_motion_weights()
    clip = torch.multinomial(w / w.sum(), n, replacement=True, generator=gen)
    lo = (W - 1) * dt
    length = lib.get_motion_lengths()[clip]
    loop = lib.get_motion_loop_mode(clip) == 1                        # the pace and trot cycles (~0.6 s) loop
    span = torch.where(loop, length, (length - lo).clamp_min(1e-4))
    t_end = lo + torch.rand(n, device=w.device, generator=gen) * span
    return mocap_windows(env, clip, t_end, W), clip


class StateRecorder:
    """Keeps the newest simulated state of every env at every step, to build AMP windows of any length."""
    KEYS = ["root_pos", "root_rot", "root_vel", "root_ang_vel", "joint_rot", "dof_vel", "body_pos"]

    def __init__(self, env, steps):
        self.env = env
        self.buf = {}
        for k in self.KEYS:
            cur = getattr(env, "_disc_hist_" + k).get_all()[:, -1]
            self.buf[k] = torch.zeros((cur.shape[0], steps) + tuple(cur.shape[1:]), device=cur.device, dtype=cur.dtype)

    def record(self, t):
        for k in self.KEYS:
            self.buf[k][:, t] = getattr(self.env, "_disc_hist_" + k).get_all()[:, -1]

    def windows(self, env_ids, t_end, W, chunk=4096):
        """(N, W, C) AMP features of the W-step windows ending at step t_end of env env_ids."""
        env = self.env
        offs = torch.arange(-(W - 1), 1, device=t_end.device)
        out = []
        for i in range(0, len(t_end), chunk):
            e, t = env_ids[i:i + chunk], t_end[i:i + chunk]
            ti = t[:, None] + offs[None, :]
            s = {k: self.buf[k][e[:, None], ti] for k in self.KEYS}
            if env._track_global_root():
                ref_pos = torch.zeros_like(s["root_pos"][:, -1])
                ref_rot = torch.zeros_like(s["root_rot"][:, -1])
                ref_rot[..., -1] = 1
            else:
                ref_pos, ref_rot = s["root_pos"][:, -1], s["root_rot"][:, -1]
            key_pos = s["body_pos"][:, :, env._key_body_ids] if env._has_key_bodies() else torch.zeros([0], device=e.device)
            obs = amp_env.compute_disc_obs(ref_root_pos=ref_pos, ref_root_rot=ref_rot, root_pos=s["root_pos"],
                                           root_rot=s["root_rot"], root_vel=s["root_vel"],
                                           root_ang_vel=s["root_ang_vel"], joint_rot=s["joint_rot"],
                                           dof_vel=s["dof_vel"], key_pos=key_pos, global_obs=env._global_obs,
                                           root_height_obs=env._root_height_obs, dof_vel_obs=env._disc_dof_vel_obs)
            out.append(obs.view(len(t), W, -1))
        return torch.cat(out)
