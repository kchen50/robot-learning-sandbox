import argparse

import torch

from policy.ddpm_policy import DiffusionPolicy
from training.imitation import train_BC_DDPM_policy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a diffusion-based BC policy for the point robot."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/2026-02-21_23-19-38-trajectories-500",
        help="Path to dataset root.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    policy = DiffusionPolicy()
    _, _, _, best_policy = train_BC_DDPM_policy(
        policy,
        dataset_root=args.data_path,
        num_epochs=args.epochs,
        batch_size_train=args.batch_size,
        lr=args.lr,
    )

    if best_policy is not None:
        policy.load_state_dict(best_policy)
    torch.save(policy.state_dict(), "./bc_ddpm_policy.pth")

