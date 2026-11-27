"""
Motion VQ-VAE with EMA codebook updates and dead-code reset (offline analysis only).

Input: windows of per-frame motion features, shape (B, C, W). Here C is the per-frame AMP discriminator
observation (root height and rotation, local root velocities, joint rotations and velocities, key-body
positions), so the codes live in the same morphology-agnostic space as the AMP-feature clustering.

  encoder     1D convs with stride 2, n_down times (temporal downsampling 2^n_down) -> z_e (B, D, T)
  quantizer   nearest of K codes (squared L2); straight-through estimator z_q = z_e + sg(e_k - z_e)
  codebook    no gradients; EMA of counts N_k and sums m_k (decay gamma), e_k = m_k / N_k
              (van den Oord et al., 2017), Laplace smoothing on N_k
  dead codes  a code whose EMA count falls below tau is re-seeded on a random z_e of the current batch
              (dead-code reset, as in MoConVQ / Jukebox)
  loss        recon MSE + beta * ||z_e - sg(z_q)||^2

Not wired to MimicKit training: nothing in mimickit/ imports this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=64, embedding_dim=64, commitment_cost=0.25, decay=0.99, epsilon=1e-5,
                 reset_threshold=1.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.reset_threshold = reset_threshold

        bound = 1.0 / num_embeddings
        embedding = torch.empty(num_embeddings, embedding_dim).uniform_(-bound, bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embedding.clone())

    def distances(self, flat_z):
        return (flat_z.pow(2).sum(1, keepdim=True) - 2 * flat_z @ self.embedding.t()
                + self.embedding.pow(2).sum(1)[None, :])

    def forward(self, z_e):
        """z_e (B, D, T) -> z_q (B, D, T), commitment loss, indices (B, T), active codes in the batch, dead resets."""
        z = z_e.permute(0, 2, 1).contiguous()
        flat_z = z.view(-1, self.embedding_dim)
        idx = self.distances(flat_z).argmin(1)
        onehot = F.one_hot(idx, self.num_embeddings).type(flat_z.dtype)

        n_dead = 0
        if self.training:
            with torch.no_grad():
                self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(onehot.t() @ flat_z, alpha=1 - self.decay)

                dead = self.cluster_size < self.reset_threshold
                n_dead = int(dead.sum().item())
                if 0 < n_dead <= flat_z.shape[0]:
                    pick = flat_z[torch.randperm(flat_z.shape[0], device=flat_z.device)[:n_dead]]
                    self.embed_avg[dead] = pick * self.reset_threshold
                    self.cluster_size[dead] = self.reset_threshold

                n = self.cluster_size.sum()
                size = (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
                self.embedding.copy_(self.embed_avg / size[:, None])

        quant = (onehot @ self.embedding).view_as(z)
        loss = self.commitment_cost * F.mse_loss(z, quant.detach())
        z_q = (z + (quant - z).detach()).permute(0, 2, 1).contiguous()
        indices = idx.view(z_e.shape[0], z_e.shape[2])
        return z_q, loss, indices, int(torch.unique(idx).numel()), n_dead


def conv_stack(c_in, hidden, c_out, n_down):
    layers, c = [], c_in
    for i in range(n_down):
        nxt = c_out if i == n_down - 1 else hidden
        layers += [nn.Conv1d(c, nxt, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.2)]
        c = nxt
    layers += [nn.Conv1d(c, c_out, kernel_size=3, padding=1)]
    return nn.Sequential(*layers)


def deconv_stack(c_in, hidden, c_out, n_down):
    layers, c = [nn.Conv1d(c_in, hidden, kernel_size=3, padding=1), nn.LeakyReLU(0.2)], hidden
    for i in range(n_down):
        layers += [nn.ConvTranspose1d(c, hidden, kernel_size=5, stride=2, padding=2, output_padding=1), nn.LeakyReLU(0.2)]
    layers += [nn.Conv1d(hidden, c_out, kernel_size=3, padding=1)]
    return nn.Sequential(*layers)


class MotionVQVAE(nn.Module):
    def __init__(self, in_channels, latent_dim=64, num_embeddings=64, hidden=128, n_down=2,
                 commitment_cost=0.25, decay=0.99, reset_threshold=1.0):
        super().__init__()
        self.config = dict(in_channels=in_channels, latent_dim=latent_dim, num_embeddings=num_embeddings,
                           hidden=hidden, n_down=n_down, commitment_cost=commitment_cost, decay=decay,
                           reset_threshold=reset_threshold)
        self.encoder = conv_stack(in_channels, hidden, latent_dim, n_down)
        self.quantizer = VectorQuantizerEMA(num_embeddings, latent_dim, commitment_cost, decay,
                                            reset_threshold=reset_threshold)
        self.decoder = deconv_stack(latent_dim, hidden, in_channels, n_down)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, commit, idx, active, n_dead = self.quantizer(z_e)
        recon = self.decoder(z_q)[..., :x.shape[-1]]
        return recon, commit, idx, active, n_dead

    @torch.no_grad()
    def encode(self, x):
        """Codes (B, T), z_e (B, D, T) and the distance of each z_e to its code (B, T)."""
        z_e = self.encoder(x)
        flat = z_e.permute(0, 2, 1).reshape(-1, z_e.shape[1])
        d = self.quantizer.distances(flat)
        dmin, idx = d.min(1)
        return idx.view(z_e.shape[0], -1), z_e, dmin.clamp_min(0).sqrt().view(z_e.shape[0], -1)

    @staticmethod
    def modal_code(indices):
        return torch.mode(indices, dim=1).values

    def code_embedding(self, k):
        return self.quantizer.embedding[k]


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = MotionVQVAE(in_channels=135).to(dev)
    x = torch.randn(16, 135, 32, device=dev)
    recon, commit, idx, active, n_dead = m(x)
    loss = F.mse_loss(recon, x) + commit
    loss.backward()
    assert recon.shape == x.shape and idx.shape == (16, 8)
    assert m.quantizer.embedding.grad is None
    print("recon", tuple(recon.shape), "codes", tuple(idx.shape), "active", active, "dead reset", n_dead,
          "modal", tuple(m.modal_code(idx).shape), "e_k", tuple(m.code_embedding(m.modal_code(idx)).shape))
