import os
import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

from utils.dataset import PointRobotDatasetManager


def get_qpos_indices(model, joints=('joint_x', 'joint_y')):
    return np.array([model.joint(j).qposadr[0] for j in joints])


def get_qvel_indices(model, joints=('joint_x', 'joint_y')):
    return np.array([model.joint(j).dofadr[0] for j in joints])


def get_ctrl_indices(model, motors=('actuator_x', 'actuator_y')):
    return np.array([model.actuator(m).id for m in motors])


def parse_args():
    p = argparse.ArgumentParser(description="Replay saved point-robot trajectories")
    p.add_argument("--dataset_root", type=str, required=True, help="Path to saved dataset folder")
    p.add_argument("--traj_index", type=int, default=0, help="Trajectory index to replay")
    p.add_argument("--xml_path", type=str, default=None, help="Override XML path (else read from meta.json)")
    p.add_argument("--loop", action="store_true", help="Loop replay until closed")
    p.add_argument("--verify", action="store_true", help="Compare simulated states to recorded states")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    return p.parse_args()


def main():
    args = parse_args()
    handle = PointRobotDatasetManager.load_root(args.dataset_root)
    meta = handle["meta"]
    xml_path = args.xml_path or meta.get("xml_path")
    if not xml_path or not os.path.exists(xml_path):
        raise FileNotFoundError("MuJoCo XML not found; provide --xml_path or ensure meta.json has xml_path")
    per_step = bool(meta.get("per_step", False))
    steps_per_action = int(meta.get("steps_per_action", 5))

    # Load one trajectory
    tds = PointRobotDatasetManager.get_trajectory_dataset(args.dataset_root, as_torch=False)
    if args.traj_index < 0 or args.traj_index >= len(tds):
        raise IndexError(f"traj_index {args.traj_index} out of range [0, {len(tds)-1}]")
    states_ref, actions_ref, T = tds[args.traj_index]

    # Build sim
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    qpos_idx = get_qpos_indices(model)
    qvel_idx = get_qvel_indices(model)
    ctrl_idx = get_ctrl_indices(model)

    def replay_once():
        mujoco.mj_resetData(model, data)
        # Initialize to first state
        data.qpos[qpos_idx] = states_ref[0][: len(qpos_idx)]
        data.qvel[qvel_idx] = states_ref[0][len(qpos_idx) : len(qpos_idx) + len(qvel_idx)]
        mujoco.mj_forward(model, data)
        err_accum = 0.0
        err_max = 0.0

        with mujoco.viewer.launch_passive(model, data) as viewer:
            t = 0

            time.sleep(1)
            
            for a in actions_ref:
                repeats = 1 if per_step else steps_per_action
                for _ in range(repeats):
                    data.ctrl[ctrl_idx] = a
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(max(0.0, model.opt.timestep / max(1e-6, args.speed)))
                if args.verify:
                    # Compare against next recorded state (node boundary for non-per_step)
                    s_next = states_ref[t + 1]
                    pose = data.qpos[qpos_idx].copy()
                    vel = data.qvel[qvel_idx].copy()
                    s_sim = np.concatenate([pose, vel])
                    e = float(np.linalg.norm(s_sim - s_next))
                    err_accum += e
                    err_max = max(err_max, e)
                t += 1
            if args.verify and t > 0:
                print(f"Avg L2 error: {err_accum / t:.6f}, Max L2 error: {err_max:.6f}")

            time.sleep(1)

    if args.loop:
        while True:
            replay_once()
    else:
        replay_once()


if __name__ == "__main__":
    main()


