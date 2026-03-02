import torch

from policy.ddpm_policy import DiffusionPolicy
from training.imitation import train_BC_DDPM_policy


if __name__ == "__main__":
    policy = DiffusionPolicy()
    data_path = "data/2026-02-21_23-01-12-trajectories-10000"
    # data_path = "data/2026-02-21_21-20-06-trajectories-2500"
    # data_path = "data/2026-02-21_23-19-38-trajectories-500"
    _, _, _, best_policy = train_BC_DDPM_policy(policy, data_path)

    if best_policy is not None:
        policy.load_state_dict(best_policy)
    torch.save(policy.state_dict(), "./bc_ddpm_policy.pth")
