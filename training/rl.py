import numpy as np
import mujoco
import time

import mujoco.viewer
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim

from policy.actor_critic_policy import Actor, Critic
from utils.mj_utils import (
    get_qpos_indices,
    get_qvel_indices,
    get_ctrl_indices,
    set_ctrl_values,
    get_qpos_values,
    get_qvel_values,
)

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


def in_goal(pose):
    return 0.8 < pose[0] < 1.1 and -0.3 < pose[1] < 0.3


def take_action(m, d, forces, enable_execution_noise: bool = False):
    qi = get_qpos_indices(m)
    vi = get_qvel_indices(m)
    ci = get_ctrl_indices(m)
    if enable_execution_noise:
        noisy_forces = forces + np.random.normal(0, 0.01, 2)
    else:
        noisy_forces = forces
    set_ctrl_values(d, ci, noisy_forces)

    delta_t = 0.1
    for _ in range(int(delta_t / m.opt.timestep)):
        mujoco.mj_step(m, d)


class RLDataset(Dataset):
    def __init__(self, states, actions, rewards, next_states):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_state": self.next_states[idx],
        }

    def add_datapoint(self, state, action, reward, next_state):
        self.states = torch.cat([self.states, state], dim=0)
        self.actions = torch.cat([self.actions, action], dim=0)
        self.rewards = torch.cat([self.rewards, reward], dim=0)
        self.next_states = torch.cat([self.next_states, next_state], dim=0)


def reward_function(state):
    distance_to_goal = np.linalg.norm(np.array([0.25, -0.1, 0.0, 0.0]) - state)
    if distance_to_goal < 0.1:
        return torch.tensor([[1.0]])
    else:
        return torch.tensor([[0.0]])


def update_networks(
    state,
    action,
    reward,
    next_state,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    gamma,
):
    with torch.no_grad():
        next_value = critic(next_state)
    value = critic(state)
    advantage = reward + gamma * next_value - value
    TD_target = reward + gamma * next_value
    print(advantage[0])

    critic_loss = torch.mean(torch.pow(value - TD_target.detach(), 2))
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    action_logprob = actor(state, action)
    actor_loss = torch.mean(-action_logprob * advantage.detach())
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return actor_loss.item(), critic_loss.item()


def _state_from_sim(d, qi, vi):
    pose = get_qpos_values(d, qi)
    vel = get_qvel_values(d, vi)
    state_np = np.concatenate((pose, vel), axis=0).astype(np.float32)
    state = torch.from_numpy(state_np).unsqueeze(0)
    return state_np, state


def _collect_rollouts(
    m, d, actor, qi, vi, episode_count, episode_length,
    real_time, viewer, enable_execution_noise,
):
    sum_rewards = 0
    full_data = None
    for ep in range(episode_count):
        mujoco.mj_resetData(m, d)
        if real_time and viewer is not None:
            viewer.sync()
        for step in range(episode_length):
            _, state = _state_from_sim(d, qi, vi)
            action, _ = actor.sample_action(state, noise=(not real_time))
            take_action(m, d, np.asarray(action), enable_execution_noise=enable_execution_noise)
            if real_time and viewer is not None:
                viewer.sync()
                time.sleep(m.opt.timestep * 2.0)
            next_state_np, next_state = _state_from_sim(d, qi, vi)
            reward = reward_function(next_state_np)
            action = action.unsqueeze(0)
            if step == 0:
                episode_data = RLDataset(state, action, reward, next_state)
            else:
                episode_data.add_datapoint(state, action, reward, next_state)
        for k in range(episode_data.rewards.shape[0] - 1, -1, -1):
            sum_rewards += float(episode_data.rewards[k])
        real_time = False
        if ep == 0:
            full_data = episode_data
        else:
            full_data.add_datapoint(
                episode_data.states,
                episode_data.actions,
                episode_data.rewards,
                episode_data.next_states,
            )
    return full_data, sum_rewards


def _train_on_batches(
    full_data, batch_size, actor, critic,
    actor_optimizer, critic_optimizer, gamma,
):
    for batch in DataLoader(full_data, batch_size=batch_size, shuffle=True):
        actor_loss, critic_loss = update_networks(
            batch["state"], batch["action"], batch["reward"], batch["next_state"],
            actor, critic, actor_optimizer, critic_optimizer, gamma,
        )
    return actor_loss, critic_loss


def train_actor_critic_policy(
    m,
    d,
    max_updates: int = 1000,
    episode_length: int = 50,
    episode_count: int = 100,
    batch_size: int = 64,
    state_dim: int = 4,
    action_dim: int = 2,
    gamma: float = 0.99,
    lr: float = 0.003,
    verbose: bool = True,
    enable_real_time: bool = True,
    enable_execution_noise: bool = False,
    actor_path: str | None = None,
    critic_path: str | None = None,
    use_viewer: bool = True,
):
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    if actor_path is not None:
        actor.load_state_dict(
            torch.load(actor_path, map_location="cpu", weights_only=True)
        )
    if critic_path is not None:
        critic.load_state_dict(
            torch.load(critic_path, map_location="cpu", weights_only=True)
        )
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

    qi = get_qpos_indices(m)
    vi = get_qvel_indices(m)
    viewer = None
    if use_viewer and enable_real_time:
        try:
            viewer = mujoco.viewer.launch_passive(m, d)
        except Exception as e:
            print(f"Viewer unavailable, continuing headless: {e}")
            viewer = None

    total_num_episodes = []
    avg_rewards = []
    update_iterable = (
        _tqdm(range(max_updates), desc="Training actor-critic", dynamic_ncols=True)
        if verbose
        else range(max_updates)
    )

    for data_collect_it in update_iterable:
        real_time = enable_real_time and (viewer is not None) and (data_collect_it % 3 == 0)
        full_data, sum_rewards = _collect_rollouts(
            m, d, actor, qi, vi, episode_count, episode_length,
            real_time, viewer, enable_execution_noise,
        )
        actor_loss, critic_loss = _train_on_batches(
            full_data, batch_size, actor, critic,
            actor_optimizer, critic_optimizer, gamma,
        )

        n_steps = episode_count * episode_length
        total_num_episodes.append((data_collect_it + 1) * episode_count)
        avg_rewards.append(sum_rewards / n_steps)
        print(f"Actor loss: {actor_loss}, Critic loss: {critic_loss}, Avg rewards: {sum_rewards / n_steps}")

        torch.save(actor.state_dict(), "actor.pth")
        torch.save(critic.state_dict(), "critic.pth")
        plt.plot(total_num_episodes, avg_rewards)
        plt.xlabel("Number of episodes")
        plt.ylabel("Average rewards")
        plt.savefig("learning_curve.png")

    return total_num_episodes, avg_rewards


__all__ = ["train_actor_critic_policy"]

