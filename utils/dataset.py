"""
Utilities for loading and saving point robot navigation datasets.
"""

import os
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# Optional torch dependency - imported lazily
def _to_tensor(x, as_torch: bool):
    if not as_torch:
        return x
    try:
        import torch  # type: ignore
    except Exception:
        raise RuntimeError("PyTorch not available but as_torch=True was requested")
    if isinstance(x, np.ndarray):
        if x.dtype.kind in ("i", "u"):
            return torch.from_numpy(x.astype(np.int64, copy=False))
        if x.dtype == np.bool_:
            return torch.from_numpy(x.astype(np.bool_, copy=False))
        return torch.from_numpy(x.astype(np.float32, copy=False))
    return x


# Optional progress bar (fallback to no-op if tqdm not installed)
try:
    from tqdm import tqdm as _tqdm
except Exception:
    class _tqdm:  # type: ignore
        def __init__(self, total=None, desc=None, dynamic_ncols=True):
            self.total = total
            self.n = 0
        def update(self, n=1):
            self.n += n
        def set_postfix_str(self, s):
            pass
        def close(self):
            pass


def _now_stamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pack_ragged(trajectories: List[Dict[str, Any]], cast_actions_fp16: bool = False) -> Dict[str, np.ndarray]:
    states_list = []
    actions_list = []
    for t in trajectories:
        s = np.asarray(t["states"])
        a = np.asarray(t["actions"])
        if s.ndim != 2 or a.ndim != 2:
            continue  # skip malformed
        if len(s) != len(a) + 1:
            continue  # require per-step alignment (states[:-1] -> actions)
        states_list.append(s.astype(np.float32, copy=False))
        if cast_actions_fp16:
            actions_list.append(a.astype(np.float16, copy=False))
        else:
            actions_list.append(a.astype(np.float32, copy=False))
    if not states_list:
        return {}
    traj_lengths = np.array([len(a) for a in actions_list], dtype=np.int32)
    traj_state_offsets = [0]
    traj_action_offsets = [0]
    for s, a in zip(states_list, actions_list):
        traj_state_offsets.append(traj_state_offsets[-1] + len(s))
        traj_action_offsets.append(traj_action_offsets[-1] + len(a))
    states = np.concatenate(states_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    dones = np.zeros(actions.shape[0], dtype=np.bool_)
    idx = 0
    for l in traj_lengths:
        if l > 0:
            idx += l
            dones[idx - 1] = True
    return {
        "states": states,
        "actions": actions,
        "dones": dones,
        "traj_state_offsets": np.array(traj_state_offsets, dtype=np.int64),
        "traj_action_offsets": np.array(traj_action_offsets, dtype=np.int64),
        "traj_lengths": traj_lengths,
    }


@dataclass
class _ShardIndex:
    path: str
    num_trajs: int
    num_steps: int
    traj_lengths: np.ndarray  # shape [N]
    # For fast mapping local step -> local traj:
    traj_action_offsets: np.ndarray  # shape [N+1]
    traj_state_offsets: np.ndarray  # shape [N+1]


class StepDataset:
    def __init__(self, root: str, as_torch: bool = False, cache_shards: int = 2):
        """
        Step-level dataset backed entirely in memory.

        All shard files are loaded eagerly so that training does not incur
        on-demand disk or mmap overhead.
        """
        self.root = root
        self.as_torch = as_torch
        self.shards_dir = os.path.join(root, "shards")
        self.meta = self._read_json(os.path.join(root, "meta.json"))
        self._indices: List[_ShardIndex] = []
        self._cumulative_steps: List[int] = []
        self._load_indices()

        # Eagerly load all shard arrays into memory
        self._cache: Dict[str, Any] = {}
        for idx in self._indices:
            with np.load(idx.path, allow_pickle=False) as z:
                self._cache[idx.path] = {
                    "states": z["states"],
                    "actions": z["actions"],
                }

    def _read_json(self, p: str) -> Dict[str, Any]:
        with open(p, "r") as f:
            return json.load(f)

    def _load_indices(self):
        shard_files = sorted(
            [f for f in os.listdir(self.shards_dir) if f.endswith(".npz")]
        )
        total = 0
        for f in shard_files:
            p = os.path.join(self.shards_dir, f)
            with np.load(p, allow_pickle=False, mmap_mode="r") as z:
                traj_lengths = z["traj_lengths"]
                ta = z["traj_action_offsets"]
                ts = z["traj_state_offsets"]
                steps = int(ta[-1])
                self._indices.append(
                    _ShardIndex(
                        path=p,
                        num_trajs=int(len(traj_lengths)),
                        num_steps=steps,
                        traj_lengths=traj_lengths,
                        traj_action_offsets=ta,
                        traj_state_offsets=ts,
                    )
                )
                total += steps
                self._cumulative_steps.append(total)

    def __len__(self) -> int:
        return self._cumulative_steps[-1] if self._cumulative_steps else 0

    def _find_shard_for_step(self, idx: int) -> Tuple[int, int]:
        # returns (shard_idx, local_step_idx)
        import bisect
        si = bisect.bisect_right(self._cumulative_steps, idx)
        prev_total = 0 if si == 0 else self._cumulative_steps[si - 1]
        return si, idx - prev_total

    # def _load_shard(self, path: str):
    #     if path in self._cache:
    #         # update LRU
    #         self._cache_order.remove(path)
    #         self._cache_order.append(path)
    #         return self._cache[path]
    #     data = np.load(path, allow_pickle=False, mmap_mode="r")
    #     self._cache[path] = data
    #     self._cache_order.append(path)
    #     if len(self._cache_order) > self._cache_limit:
    #         old = self._cache_order.pop(0)
    #         try:
    #             self._cache.pop(old, None)
    #         except Exception:
    #             pass
    #     return data

    def _load_shard(self, path: str):
        # All shards are loaded eagerly in __init__, so this is just a dict lookup.
        return self._cache[path]

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError
        shard_idx, local_step = self._find_shard_for_step(idx)
        shard = self._indices[shard_idx]
        z = self._load_shard(shard.path)
        # find local trajectory via action offsets
        import bisect
        t_local = bisect.bisect_right(shard.traj_action_offsets, local_step) - 1
        a0 = int(shard.traj_action_offsets[t_local])
        s0 = int(shard.traj_state_offsets[t_local])
        within = local_step - a0
        # state at time t (align with a_t)
        state = z["states"][s0 + within]
        action = z["actions"][a0 + within]
        return _to_tensor(state, self.as_torch), _to_tensor(action, self.as_torch)


class TrajectoryDataset:
    def __init__(self, root: str, as_torch: bool = False, cache_shards: int = 2):
        """
        Trajectory-level dataset backed entirely in memory.

        All shard files are loaded eagerly so that iteration over full
        trajectories does not incur on-demand disk or mmap overhead.
        """
        self.root = root
        self.as_torch = as_torch
        self.shards_dir = os.path.join(root, "shards")
        self.meta = self._read_json(os.path.join(root, "meta.json"))
        self._indices: List[_ShardIndex] = []
        self._cumulative_trajs: List[int] = []
        self._load_indices()

        # Eagerly load all shard arrays into memory
        self._cache: Dict[str, Any] = {}
        for idx in self._indices:
            with np.load(idx.path, allow_pickle=False) as z:
                self._cache[idx.path] = {
                    "states": z["states"],
                    "actions": z["actions"],
                }

    def _read_json(self, p: str) -> Dict[str, Any]:
        with open(p, "r") as f:
            return json.load(f)

    def _load_indices(self):
        shard_files = sorted(
            [f for f in os.listdir(self.shards_dir) if f.endswith(".npz")]
        )
        total = 0
        for f in shard_files:
            p = os.path.join(self.shards_dir, f)
            with np.load(p, allow_pickle=False, mmap_mode="r") as z:
                traj_lengths = z["traj_lengths"]
                ta = z["traj_action_offsets"]
                ts = z["traj_state_offsets"]
                n = int(len(traj_lengths))
                self._indices.append(
                    _ShardIndex(
                        path=p,
                        num_trajs=n,
                        num_steps=int(ta[-1]),
                        traj_lengths=traj_lengths,
                        traj_action_offsets=ta,
                        traj_state_offsets=ts,
                    )
                )
                total += n
                self._cumulative_trajs.append(total)

    def __len__(self) -> int:
        return self._cumulative_trajs[-1] if self._cumulative_trajs else 0

    def _find_shard_for_traj(self, idx: int) -> Tuple[int, int]:
        import bisect
        si = bisect.bisect_right(self._cumulative_trajs, idx)
        prev_total = 0 if si == 0 else self._cumulative_trajs[si - 1]
        return si, idx - prev_total

    def _load_shard(self, path: str):
        # All shards are loaded eagerly in __init__, so this is just a dict lookup.
        return self._cache[path]

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError
        shard_idx, local_traj = self._find_shard_for_traj(idx)
        shard = self._indices[shard_idx]
        z = self._load_shard(shard.path)
        s0 = int(shard.traj_state_offsets[local_traj])
        s1 = int(shard.traj_state_offsets[local_traj + 1])
        a0 = int(shard.traj_action_offsets[local_traj])
        a1 = int(shard.traj_action_offsets[local_traj + 1])
        states = z["states"][s0:s1]
        actions = z["actions"][a0:a1]
        return _to_tensor(states, self.as_torch), _to_tensor(actions, self.as_torch), (s1 - s0 - 1)


class PointRobotDatasetManager:
    @staticmethod
    def save(
        trajectories: List[Dict[str, Any]],
        out_root: Optional[str] = None,
        shard_size: int = 500,
        cast_actions_fp16: bool = False,
        with_splits: Optional[Tuple[float, float, float]] = None,  # e.g., (0.7, 0.15, 0.15)
        meta_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        assert shard_size > 0
        trajs = [t for t in trajectories if isinstance(t, dict) and "states" in t and "actions" in t]
        num_traj = len(trajs)
        ts = _now_stamp()
        root = out_root or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", f"{ts}-trajectories-{num_traj}"
        )
        shards_dir = os.path.join(root, "shards")
        _ensure_dir(shards_dir)

        # Write shards
        pbar = _tqdm(total=num_traj, desc="Saving shards", dynamic_ncols=True)
        num_shards = (num_traj + shard_size - 1) // shard_size
        shard_paths = []
        traj_cursor = 0
        for shard_idx in range(num_shards):
            batch = trajs[shard_idx * shard_size : (shard_idx + 1) * shard_size]
            packed = _pack_ragged(batch, cast_actions_fp16=cast_actions_fp16)
            if not packed:
                continue
            shard_path = os.path.join(shards_dir, f"part-{shard_idx:05d}.npz")
            np.savez_compressed(
                shard_path,
                states=packed["states"],
                actions=packed["actions"],
                dones=packed["dones"],
                traj_state_offsets=packed["traj_state_offsets"],
                traj_action_offsets=packed["traj_action_offsets"],
                traj_lengths=packed["traj_lengths"],
            )
            shard_paths.append(os.path.basename(shard_path))
            traj_cursor += len(batch)
            pbar.update(len(batch))
            pbar.set_postfix_str(f"shard {shard_idx+1}/{num_shards}")
        pbar.close()

        # Meta with replay hints (future-proofing)
        # Infer dims if available in the last packed shard
        inferred_state_dim = None
        inferred_action_dim = None
        try:
            if shard_paths:
                with np.load(os.path.join(shards_dir, shard_paths[-1]), allow_pickle=False, mmap_mode="r") as z:
                    inferred_state_dim = int(z["states"].shape[1])
                    inferred_action_dim = int(z["actions"].shape[1])
        except Exception:
            pass

        meta = {
            "created_at": ts,
            "num_trajectories": num_traj,
            "shard_size": shard_size,
            "num_shards": len(shard_paths),
            "dtype_actions": "float16" if cast_actions_fp16 else "float32",
            # Replayability: if saving states+actions per-step, data is fully replayable
            "replayable": True,
            "fields": ["states", "actions", "dones", "traj_state_offsets", "traj_action_offsets", "traj_lengths"],
            "shards": shard_paths,
            "state_dim": inferred_state_dim,
            "action_dim": inferred_action_dim,
        }
        if meta_overrides:
            meta.update(dict[str, Any](meta_overrides))
        _ensure_dir(root)
        with open(os.path.join(root, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Optional splits by shard (coarse)
        if with_splits:
            train_p, val_p, test_p = with_splits
            total = len(shard_paths)
            n_train = int(round(train_p * total))
            n_val = int(round(val_p * total))
            n_test = total - n_train - n_val
            splits_dir = os.path.join(root, "splits")
            _ensure_dir(splits_dir)
            def _write_split(name, items):
                with open(os.path.join(splits_dir, f"{name}.txt"), "w") as f:
                    for it in items:
                        f.write(f"{it}\n")
            _write_split("train", shard_paths[:n_train])
            _write_split("val", shard_paths[n_train:n_train + n_val])
            _write_split("test", shard_paths[n_train + n_val:])

        return root

    @staticmethod
    def load_root(saved_path: str) -> Dict[str, Any]:
        meta_path = os.path.join(saved_path, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        shards_dir = os.path.join(saved_path, "shards")
        return {"meta": meta, "shards_dir": shards_dir}

    @staticmethod
    def get_step_dataset(saved_path: str, as_torch: bool = False) -> StepDataset:
        return StepDataset(saved_path, as_torch=as_torch)

    @staticmethod
    def get_trajectory_dataset(saved_path: str, as_torch: bool = False) -> TrajectoryDataset:
        return TrajectoryDataset(saved_path, as_torch=as_torch)


