import os
import argparse
import json
from typing import Optional

from planner.rrt_planner import RRTPlanner
from planner.rrt_multi import collect_parallel_chunks

from utils.dataset import PointRobotDatasetManager

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect point-robot trajectories and save as a dataset")
    default_xml = os.path.join(os.path.dirname(__file__), "scenes/point_robot_nav.xml")
    p.add_argument("--xml_path", type=str, default=default_xml, help="Path to MuJoCo XML")
    p.add_argument("--num_trajectories", type=int, default=5, help="Total trajectories to collect")
    p.add_argument("--per_step", action="store_true", help="Log per-physics-step (s,a) samples")
    p.add_argument("--steps_per_action", type=int, default=5, help="Planner steps per action")
    p.add_argument("--time_limit", type=float, default=30.0, help="Time limit per planning attempt (seconds)")
    p.add_argument("--num_workers", type=int, default=1, help="Processes for parallel collection (>=2 uses multiprocess)")
    p.add_argument("--chunk_size", type=int, default=5, help="Trajectories per task in multiprocess mode")
    p.add_argument("--shard_size", type=int, default=500, help="Trajectories per saved shard (.npz)")
    p.add_argument("--seed", type=int, default=123, help="Base RNG seed")
    p.add_argument("--kdtree_rebuild_every", type=int, default=64, help="KDTree rebuild cadence")
    p.add_argument("--randomize_start", action="store_true", help="Randomize start poses")
    p.add_argument("--out_root", type=str, default=None, help="Optional output root directory")
    p.add_argument("--cast_actions_fp16", action="store_true", help="Store actions as float16 to reduce size")
    p.add_argument("--splits", type=str, default=None, help="Optional splits as JSON, e.g., '{\"train\":0.8,\"val\":0.1,\"test\":0.1}'")
    p.add_argument("--config", type=str, default=None, help="Optional JSON config to fill defaults")
    # Capture defaults for selective override
    defaults = {a.dest: a.default for a in p._actions if a.dest != "help"}
    args = p.parse_args()
    # Config file overrides only fields left at default
    if args.config:
        try:
            with open(args.config, "r") as f:
                cfg = json.load(f)
            for k, v in cfg.items():
                if hasattr(args, k) and getattr(args, k) == defaults.get(k):
                    setattr(args, k, v)
        except Exception:
            pass
    # Env fallback for xml path
    if not args.xml_path or not os.path.exists(args.xml_path):
        env_xml = os.getenv("PR_XML_PATH")
        if env_xml and os.path.exists(env_xml):
            args.xml_path = env_xml
    return args


def main():
    args = parse_args()

    # Collect trajectories
    if args.num_workers is None or args.num_workers <= 1:
        planner = RRTPlanner(
            xml_path=args.xml_path,
            steps_per_action=args.steps_per_action,
            time_limit_seconds=args.time_limit,
            kdtree_rebuild_every=args.kdtree_rebuild_every,
        )
        trajs = planner.collect(
            num_trajectories=args.num_trajectories,
            seed=args.seed,
            randomize_start=args.randomize_start,
            min_plan_len=1,
            show_progress=True,
            per_step=args.per_step,
        )
        collection_mode = "single"
    else:
        # Multiprocess path (per_step currently not threaded; collects node-level unless planner default changed)
        trajs = collect_parallel_chunks(
            xml_path=args.xml_path,
            total_trajectories=args.num_trajectories,
            chunk_size=args.chunk_size,
            num_workers=args.num_workers,
            base_seed=args.seed,
            steps_per_action=args.steps_per_action,
            time_limit_seconds=args.time_limit,
            kdtree_rebuild_every=args.kdtree_rebuild_every,
            randomize_start=args.randomize_start,
            min_plan_len=1,
            verbose=True,
        )
        collection_mode = "multiprocess"

    # Prepare meta overrides (replay-friendly)
    # Infer dims from first trajectory if available
    state_dim = int(trajs[0]["states"].shape[1]) if len(trajs) > 0 else None
    action_dim = int(trajs[0]["actions"].shape[1]) if len(trajs) > 0 else None
    meta_overrides = {
        "xml_path": args.xml_path,
        "steps_per_action": args.steps_per_action,
        "per_step": bool(args.per_step),
        "time_limit_seconds": args.time_limit,
        "kdtree_rebuild_every": args.kdtree_rebuild_every,
        "randomize_start": bool(args.randomize_start),
        "base_seed": int(args.seed),
        "collection_mode": collection_mode,
        "num_workers": int(args.num_workers),
        "chunk_size": int(args.chunk_size),
        "total_trajectories": int(args.num_trajectories),
        "state_dim": state_dim,
        "action_dim": action_dim,
    }
    with_splits = None
    if args.splits:
        try:
            s = json.loads(args.splits)
            train = float(s.get("train", 0.8))
            val = float(s.get("val", 0.1))
            test = float(s.get("test", 0.1))
            with_splits = (train, val, test)
        except Exception:
            with_splits = None

    saved_path = PointRobotDatasetManager.save(
        trajectories=trajs,
        out_root=args.out_root,
        shard_size=args.shard_size,
        cast_actions_fp16=args.cast_actions_fp16,
        with_splits=with_splits,
        meta_overrides=meta_overrides,
    )
    print(saved_path)


if __name__ == "__main__":
    main()


