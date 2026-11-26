"""
AMP agent for the gait-conditioned env (gait_cond_steering_env).

The condition (one-hot gait) is already part of the policy and discriminator observations, so the
policy, critic and discriminator networks are Peng's AMP model unchanged. The only difference is the
discriminator gradient penalty: it is computed on the motion part of the input only. Penalizing the
gradient w.r.t. the one-hot would push the discriminator to ignore the condition, which is exactly
the signal the conditioning needs.

Extra log entries: per gait class, the mean discriminator logit of the agent windows
(Disc_Agent_Logit_<gait>), to see whether one commanded gait stays far from its reference.
"""

import torch

import learning.amp_agent as amp_agent


class GaitCondAMPAgent(amp_agent.AMPAgent):
    def __init__(self, config, env, device):
        self._cond_dim = env.get_disc_cond_dim()
        self._gait_names = env.get_gait_classes()
        super().__init__(config, env, device)
        return

    def _grad_sq(self, logit, inputs):
        grad = torch.autograd.grad(logit, inputs, grad_outputs=torch.ones_like(logit),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad = grad[..., :-self._cond_dim]           # motion features only
        return torch.sum(torch.square(grad), dim=-1)

    def _compute_disc_loss(self, batch):
        disc_obs = batch["disc_obs"]
        disc_demo_obs = batch["disc_obs_demo"]

        norm_disc_obs_demo = self._disc_obs_norm.normalize(disc_demo_obs)
        norm_disc_obs_demo.requires_grad_(True)

        replay_data = self._disc_buffer.sample(disc_obs.shape[0])
        replay_obs = replay_data["disc_obs"]
        disc_obs = torch.cat([disc_obs, replay_obs], dim=0)
        norm_disc_obs = self._disc_obs_norm.normalize(disc_obs)
        norm_disc_obs.requires_grad_(True)

        disc_agent_logit = self._model.eval_disc(norm_disc_obs).squeeze(-1)
        disc_demo_logit = self._model.eval_disc(norm_disc_obs_demo).squeeze(-1)

        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        disc_demo_grad = self._grad_sq(disc_demo_logit, norm_disc_obs_demo)
        disc_agent_grad = self._grad_sq(disc_agent_logit, norm_disc_obs)
        disc_grad_penalty = 0.5 * (torch.mean(disc_demo_grad) + torch.mean(disc_agent_grad))
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        disc_info = {
            "disc_loss": disc_loss,
            "disc_grad_penalty": disc_grad_penalty.detach(),
            "disc_agent_acc": disc_agent_acc.detach(),
            "disc_demo_acc": disc_demo_acc.detach(),
            "disc_agent_logit": torch.mean(disc_agent_logit).detach(),
            "disc_demo_logit": torch.mean(disc_demo_logit).detach()
        }

        # mean agent logit per commanded gait (one-hot = last cond_dim raw entries)
        agent_gait = torch.argmax(disc_obs[..., -self._cond_dim:], dim=-1)
        for k, g in enumerate(self._gait_names):
            m = (agent_gait == k).float()
            disc_info["disc_agent_logit_" + g] = ((disc_agent_logit.detach() * m).sum() / m.sum().clamp_min(1.0))

        if (self._disc_logit_reg != 0):
            logit_weights = self._model.get_disc_logit_weights()
            disc_logit_loss = torch.sum(torch.square(logit_weights))
            disc_loss += self._disc_logit_reg * disc_logit_loss
            disc_info["disc_logit_loss"] = disc_logit_loss.detach()

        return disc_info
