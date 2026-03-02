import torch
import torch.nn as nn

from policy.policy import STATE_DIM, ACTION_DIM


class DiffusionPolicy(nn.Module):
    """
    Simple MLP-based conditional diffusion model over actions given state.
    Forward signature matches training usage:
        eps_pred = policy(states, noisy_actions, timesteps)
    where:
        states:        (B, STATE_DIM)
        noisy_actions: (B, ACTION_DIM)
        timesteps:     (B,) integer indices in [0, T-1]
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 64,
        num_diffusion_steps: int = 50,
        time_emb_dim: int = 32,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps

        self.time_embed = nn.Embedding(num_diffusion_steps, time_emb_dim)

        in_dim = state_dim + action_dim + time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states, noisy_actions, timesteps):
        """
        Predict noise given state, noisy action, and diffusion timestep.
        """
        if timesteps.dtype != torch.long:
            timesteps = timesteps.long()
        t_emb = self.time_embed(timesteps)
        x = torch.cat([states, noisy_actions, t_emb], dim=-1)
        eps_pred = self.net(x)
        return eps_pred

    def _schedule(self, device):
        T = self.num_diffusion_steps
        betas = torch.linspace(1e-4, 2e-2, T, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars

    @torch.no_grad()
    def sample(self, state: torch.Tensor, num_steps: int = None) -> torch.Tensor:
        """
        Sample an action by running the DDPM reverse process conditioned on state.
        state: (B, state_dim) or (state_dim,). Returns (B, action_dim) or (action_dim,).
        """
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)
        device = state.device
        T = num_steps if num_steps is not None else self.num_diffusion_steps
        betas, alphas, alpha_bars = self._schedule(device)
        B = state.shape[0]
        a = torch.randn(B, self.action_dim, device=device)
        for t in range(T - 1, -1, -1):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            eps_pred = self.forward(state, a, t_batch)
            alpha_t = alphas[t].view(1, 1)
            alpha_bar_t = alpha_bars[t].view(1, 1)
            alpha_bar_t_prev = alpha_bars[t - 1].view(1, 1) if t > 0 else torch.ones(1, 1, device=device)
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                a - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * eps_pred
            )
            if t > 0:
                sigma_t = torch.sqrt(betas[t])
                a = mean + sigma_t * torch.randn_like(a, device=device)
            else:
                a = mean
        a = torch.clamp(a, -1.0, 1.0)
        return a.squeeze(0) if squeeze else a

    @staticmethod
    def load(path: str) -> "DiffusionPolicy":
        policy = DiffusionPolicy()
        policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return policy

