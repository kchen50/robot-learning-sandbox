import argparse

import mujoco

from training.rl import train_actor_critic_policy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an actor-critic policy for the point robot."
    )
    parser.add_argument(
        "--disable-real-time",
        action="store_true",
        default=False,
        help="Disable real-time visualization and viewer sync (default: False).",
    )
    parser.add_argument(
        "--enable-execution-noise",
        action="store_true",
        default=False,
        help="Add Gaussian noise to the executed actions (default: False).",
    )
    parser.add_argument(
        "--actor-path",
        type=str,
        default=None,
        help="Path to actor state dict to load (optional).",
    )
    parser.add_argument(
        "--critic-path",
        type=str,
        default=None,
        help="Path to critic state dict to load (optional).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    m = mujoco.MjModel.from_xml_path('./scenes/point_robot_nav.xml')
    d = mujoco.MjData(m)
    train_actor_critic_policy(
        m,
        d,
        enable_real_time=not args.disable_real_time,
        enable_execution_noise=args.enable_execution_noise,
        actor_path=args.actor_path,
        critic_path=args.critic_path,
    )


if __name__ == '__main__':
    main()
