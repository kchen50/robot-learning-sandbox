import copy
from typing import Optional

import torch
import torch.nn as nn

from utils.pytorch import make_step_dataloaders

try:
    from tqdm import tqdm as _tqdm
except Exception:
    class _tqdm:  # type: ignore
        def __init__(self, iterable=None, total=None, desc=None, dynamic_ncols=True):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())

        def update(self, n=1):
            pass

        def set_postfix_str(self, s):
            pass

        def close(self):
            pass


def train_BC_policy(
    policy: nn.Module,
    dataset_root: str,
    num_epochs: int = 100,
    batch_size_train: int = 1024,
    lr: float = 1e-3,
    device: Optional[str] = None,
    patience: int = 5,
    min_delta: float = 0.0,
    verbose: bool = True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(device)

    train_loader, val_loader, _ = make_step_dataloaders(
        root=dataset_root,
        batch_size_train=batch_size_train,
        ratios=(0.8, 0.1, 0.1),
        seed=42,
        num_workers_train=2,
        pin_memory=True,
        persistent_workers=True,
        full_batch_val=True,
        full_batch_test=True,
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_val_epoch = -1
    best_policy = None
    bad_epochs = 0

    for epoch in range(num_epochs):
        policy.train()
        epoch_train_losses = []
        loader_iter = (
            _tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                dynamic_ncols=True,
            )
            if verbose
            else train_loader
        )
        for states, actions in loader_iter:
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            actions_pred = policy(states)
            train_loss = loss_fn(actions_pred, actions)
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            optimizer.step()
            epoch_train_losses.append(train_loss.item())
        train_losses.append(sum(epoch_train_losses) / max(1, len(epoch_train_losses)))

        if val_loader is not None:
            policy.eval()
            with torch.no_grad():
                epoch_val_losses = []
                for states_val, actions_val in val_loader:
                    states_val = states_val.to(device)
                    actions_val = actions_val.to(device)
                    val_actions_pred = policy(states_val)
                    val_loss = loss_fn(val_actions_pred, actions_val)
                    epoch_val_losses.append(val_loss.item())
                mean_val = sum(epoch_val_losses) / max(1, len(epoch_val_losses))
                val_losses.append(mean_val)
                if verbose:
                    print(
                        f"Epoch {epoch+1}: train {train_losses[-1]:.6f} | "
                        f"val {mean_val:.6f}"
                    )
                if mean_val + min_delta < best_val_loss:
                    best_val_loss = mean_val
                    best_val_epoch = epoch
                    best_policy = copy.deepcopy(policy.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if patience > 0 and bad_epochs >= patience:
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch+1} "
                            f"(best epoch {best_val_epoch+1}, best val {best_val_loss:.6f})"
                        )
                    break
        else:
            val_losses.append(float("nan"))

    if best_policy is not None:
        policy.load_state_dict(best_policy)
    return train_losses, val_losses, best_val_epoch, best_policy


def _make_beta_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: torch.device | str = "cpu",
):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def train_BC_DDPM_policy(
    policy: nn.Module,
    dataset_root: str,
    num_epochs: int = 100,
    batch_size_train: int = 1024,
    lr: float = 1e-3,
    device: Optional[str] = None,
    patience: int = 5,
    min_delta: float = 0.0,
    verbose: bool = True,
    num_diffusion_steps: int = 50,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    policy = policy.to(dev)

    if hasattr(policy, "num_diffusion_steps"):
        num_diffusion_steps = int(getattr(policy, "num_diffusion_steps"))

    train_loader, val_loader, _ = make_step_dataloaders(
        root=dataset_root,
        batch_size_train=batch_size_train,
        ratios=(0.8, 0.1, 0.1),
        seed=42,
        num_workers_train=2,
        pin_memory=True,
        persistent_workers=True,
        full_batch_val=True,
        full_batch_test=True,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    betas, alphas, alpha_bars = _make_beta_schedule(num_diffusion_steps, device=dev)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_val_epoch = -1
    best_policy = None
    bad_epochs = 0

    for epoch in range(num_epochs):
        policy.train()
        epoch_train_losses = []
        loader_iter = (
            _tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                dynamic_ncols=True,
            )
            if verbose
            else train_loader
        )
        for states, actions in loader_iter:
            states = states.to(dev, non_blocking=True)
            actions = actions.to(dev, non_blocking=True)

            B = actions.shape[0]
            t = torch.randint(0, num_diffusion_steps, (B,), device=dev)
            alpha_bar_t = alpha_bars[t].view(B, 1)

            eps = torch.randn_like(actions)
            noisy_actions = torch.sqrt(alpha_bar_t) * actions + torch.sqrt(
                1.0 - alpha_bar_t
            ) * eps

            eps_pred = policy(states, noisy_actions, t)
            loss = mse_loss(eps_pred, eps)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())

        train_losses.append(sum(epoch_train_losses) / max(1, len(epoch_train_losses)))

        if val_loader is not None:
            policy.eval()
            with torch.no_grad():
                epoch_val_losses = []
                for states_val, actions_val in val_loader:
                    states_val = states_val.to(dev, non_blocking=True)
                    actions_val = actions_val.to(dev, non_blocking=True)

                    Bv = actions_val.shape[0]
                    t_val = torch.randint(0, num_diffusion_steps, (Bv,), device=dev)
                    alpha_bar_t_val = alpha_bars[t_val].view(Bv, 1)

                    eps_val = torch.randn_like(actions_val)
                    noisy_actions_val = torch.sqrt(alpha_bar_t_val) * actions_val + torch.sqrt(
                        1.0 - alpha_bar_t_val
                    ) * eps_val

                    eps_pred_val = policy(states_val, noisy_actions_val, t_val)
                    val_loss = mse_loss(eps_pred_val, eps_val)
                    epoch_val_losses.append(val_loss.item())

                mean_val = sum(epoch_val_losses) / max(1, len(epoch_val_losses))
                val_losses.append(mean_val)
                if verbose:
                    print(
                        f"Epoch {epoch+1}: train {train_losses[-1]:.6f} | "
                        f"val {mean_val:.6f}"
                    )

                if mean_val + min_delta < best_val_loss:
                    best_val_loss = mean_val
                    best_val_epoch = epoch
                    best_policy = copy.deepcopy(policy.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if patience > 0 and bad_epochs >= patience:
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch+1} "
                            f"(best epoch {best_val_epoch+1}, best val {best_val_loss:.6f})"
                        )
                    break
        else:
            val_losses.append(float("nan"))

    if best_policy is not None:
        policy.load_state_dict(best_policy)
    return train_losses, val_losses, best_val_epoch, best_policy


__all__ = ["train_BC_policy", "train_BC_DDPM_policy"]

