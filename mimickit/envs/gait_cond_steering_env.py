"""
Steering env conditioned on a gait label (first DCAMP test: one-hot conditioning).

Each episode has a commanded gait class c (one-hot). The one-hot is appended to
  - the policy observation, so the policy knows which gait is asked for;
  - the discriminator observation: reference windows carry the class of the clip they come from,
    the agent windows carry the commanded class.
The discriminator then models p(motion | c) instead of p(motion), so walking when c = run is
"fake" even though walking itself is in the dataset: the policy can no longer satisfy the style
reward with one gait for every command.

c is the class of the clip drawn by reference state initialization at reset, so the episode starts
inside the commanded gait and the commanded classes follow the dataset weights. The target speed is
drawn from the speed range of c (gait_speed_ranges), so the task does not ask for a run at walking
speed. c stays fixed for the episode.

For evaluation, set_forced_gait(k) makes every reset use class k (clips of that class only).
"""

import torch

import envs.task_steering_env as task_steering_env


class GaitCondSteeringEnv(task_steering_env.TaskSteeringEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        # gait_classes: ordered list of class names; gait_clip_classes: substring of the clip file -> class
        self._gait_classes = list(env_config["gait_classes"])
        self._gait_clip_classes = dict(env_config["gait_clip_classes"])
        ranges = env_config["gait_speed_ranges"]
        self._gait_speed_lo = torch.tensor([float(ranges[g][0]) for g in self._gait_classes], device=device)
        self._gait_speed_hi = torch.tensor([float(ranges[g][1]) for g in self._gait_classes], device=device)
        self._forced_gait = None
        # manual control: keys 1..K pick the commanded gait (and jump the speed to the middle of its range)
        self._manual_gait = self._gait_classes.index(env_config.get("manual_init_gait", self._gait_classes[0]))
        self._gait_keys_down = [False] * len(self._gait_classes)

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return

    # ------------------------------------------------------------------ public

    def get_num_gaits(self):
        return len(self._gait_classes)

    def get_gait_classes(self):
        return list(self._gait_classes)

    def get_disc_cond_dim(self):
        """Number of trailing disc obs entries that are the condition (not motion)."""
        return self.get_num_gaits()

    def get_gait_cmd(self):
        return self._gait_cmd

    def set_forced_gait(self, k):
        """None = sample with the dataset weights (training); an int = every reset uses that class."""
        self._forced_gait = k
        return

    # ------------------------------------------------------------------ setup

    def _load_motions(self, motion_file):
        super()._load_motions(motion_file)

        ml = self._motion_lib
        clip_cls = []
        for m in range(ml.get_num_motions()):
            f = ml.get_motion_file(m).replace("\\", "/").split("/")[-1]
            matches = [c for s, c in self._gait_clip_classes.items() if s in f]
            assert len(matches) == 1, "clip {} matches gait classes {}".format(f, matches)
            clip_cls.append(self._gait_classes.index(matches[0]))
        self._clip_gait = torch.tensor(clip_cls, device=self._device, dtype=torch.long)

        print("Gait conditioning: " + ", ".join(
            "{}={}".format(f.replace("\\", "/").split("/")[-1], self._gait_classes[c])
            for f, c in zip([ml.get_motion_file(m) for m in range(ml.get_num_motions())], clip_cls)))
        return

    def _build_sim_tensors(self, config):
        num_envs = self.get_num_envs()
        self._gait_cmd = torch.zeros(num_envs, device=self._device, dtype=torch.long)
        super()._build_sim_tensors(config)
        return

    def _build_disc_obs_buffers(self):
        # the parent allocates _disc_obs_buf with the full size (motion + one-hot), since
        # get_disc_obs_space goes through _compute_disc_obs_demo; keep it as the full buffer and
        # give the parent a motion-only buffer to fill
        super()._build_disc_obs_buffers()
        self._disc_obs_full_buf = self._disc_obs_buf
        n = self._disc_obs_full_buf.shape[-1] - self.get_disc_cond_dim()
        self._disc_obs_buf = torch.zeros([self.get_num_envs(), n], device=self._device,
                                         dtype=self._disc_obs_full_buf.dtype)
        return

    def _build_data_buffers(self):
        super()._build_data_buffers()
        self._info["disc_obs"] = self._disc_obs_full_buf
        return

    # ------------------------------------------------------------------ condition

    def _one_hot(self, k):
        return torch.nn.functional.one_hot(k, self.get_num_gaits()).to(torch.float)

    def _sample_motion_times(self, n):
        if (self._forced_gait is None):
            return super()._sample_motion_times(n)

        # clips of the forced class only, with their relative dataset weights
        w = self._motion_lib.get_motion_weights() * (self._clip_gait == self._forced_gait).float()
        motion_ids = torch.multinomial(w, num_samples=n, replacement=True)
        if (self._rand_reset):
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(n, dtype=torch.float, device=self._device)
        return motion_ids, motion_times

    def _reset_ref_motion(self, env_ids):
        super()._reset_ref_motion(env_ids)
        if (self._manual_control):
            self._gait_cmd[env_ids] = self._manual_gait
        else:
            self._gait_cmd[env_ids] = self._clip_gait[self._motion_ids[env_ids]]
        return

    # ------------------------------------------------------------------ manual gait keys

    def _setup_gui(self):
        super()._setup_gui()
        if (self._manual_control):
            for k in range(self.get_num_gaits()):
                self._engine.reserve_key(str(k + 1))
            print("Gait command: " + ", ".join("{} = {}".format(k + 1, g) for k, g in enumerate(self._gait_classes)) +
                  "  (selecting a gait sets the speed to the middle of its range; W/S still adjust it)")
            self._print_gait_cmd()
        return

    def _print_gait_cmd(self):
        k = self._manual_gait
        print("Gait: {} (speed range {:.1f}-{:.1f} m/s), speed {:.2f} m/s".format(
            self._gait_classes[k], self._gait_speed_lo[k].item(), self._gait_speed_hi[k].item(), self._manual_speed))
        return

    def _poll_manual_keys(self):
        super()._poll_manual_keys()
        for k in range(self.get_num_gaits()):
            down = self._engine.is_key_down(str(k + 1))
            if (down and not self._gait_keys_down[k]):
                self._manual_gait = k
                self._manual_speed = 0.5 * (self._gait_speed_lo[k].item() + self._gait_speed_hi[k].item())
                self._gait_cmd[:] = k
                self._print_gait_cmd()
            self._gait_keys_down[k] = down
        return

    def _reset_tar(self, env_ids):
        super()._reset_tar(env_ids)
        if (self._manual_control):
            return

        # target speed within the range of the commanded gait
        c = self._gait_cmd[env_ids]
        lo, hi = self._gait_speed_lo[c], self._gait_speed_hi[c]
        self._tar_speed[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self._device)
        return

    # ------------------------------------------------------------------ observations

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        c = self._gait_cmd if env_ids is None else self._gait_cmd[env_ids]
        return torch.cat([obs, self._one_hot(c)], dim=-1)

    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        disc_obs = super()._compute_disc_obs_demo(motion_ids, motion_times0)
        return torch.cat([disc_obs, self._one_hot(self._clip_gait[motion_ids])], dim=-1)

    def _update_disc_obs(self, env_ids=None):
        super()._update_disc_obs(env_ids)   # fills the motion-only _disc_obs_buf
        if (env_ids is None):
            self._disc_obs_full_buf[:] = torch.cat([self._disc_obs_buf, self._one_hot(self._gait_cmd)], dim=-1)
        else:
            self._disc_obs_full_buf[env_ids] = torch.cat([self._disc_obs_buf[env_ids],
                                                          self._one_hot(self._gait_cmd[env_ids])], dim=-1)
        return
