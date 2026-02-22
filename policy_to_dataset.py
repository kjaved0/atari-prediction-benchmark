#!/usr/bin/env python3
"""
Replay policy trace JSON files in ALE and dump a static dataset (images, rewards,
terminals, actions, returns) to a single .npz file.

Policy files are the existing JSON traces (e.g. policies/rainbow_Pong_eps_0.1.json)
with keys: env, test_seed, ale_state, sequence_of_actions ('a'–'z' = actions, 'R' = reset).

Returns are always stored: discounted sum of future rewards (--gamma, default 0.99).
At terminal steps, future rewards are zero. When --steps is set, the script collects
5000 extra steps per file for return computation but stores only the first --steps.

If multiple policy files are given, they are replayed in order and concatenated.
If the files use different games (different env), observations will be from
different games; use a single game per run if you want a homogeneous dataset.

Usage:
  python policy_to_dataset.py --output dataset.npz policy1.json policy2.json
  python policy_to_dataset.py --steps 100 --gamma 0.99 --output out.npz policy.json
"""

import argparse
import base64
import json
import pickle
import re
import sys
from typing import Generator, List, Optional, Tuple

import numpy as np
from ale_py import ALEInterface, roms

RETURNS_BUFFER_STEPS = 5000


def _env_to_rom_id(env: str) -> str:
    """Derive ale-py ROM id from env string (e.g. PongNoFrameskip-v4 -> pong)."""
    rom_name = env.split("-v")[0].replace("NoFrameskip", "").replace("Deterministic", "")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", rom_name).lower()


def replay_trace(policy_path: str) -> Generator[Tuple[np.ndarray, float, bool, int], None, None]:
    """
    Load one policy JSON, set up ALE, and yield (observation, reward, terminal, action_index)
    for each step. On 'R' in the action sequence we reset and do not yield.
    action_index is in 0..num_actions-1 (minimal action set).
    """
    with open(policy_path) as f:
        trace = json.load(f)

    rom_id = _env_to_rom_id(trace["env"])
    rom_path = roms.get_rom_path(rom_id)

    ale = ALEInterface()
    ale.setInt("random_seed", int(trace["test_seed"]))
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setInt("frame_skip", 1)
    ale.loadROM(rom_path)
    ale.reset_game()
    ale.restoreState(pickle.loads(base64.b64decode(trace["ale_state"])))

    actions = list(ale.getMinimalActionSet())
    seq = trace["sequence_of_actions"]

    for i, c in enumerate(seq):
        if c == "R":
            ale.reset_game()
            continue
        action_index = ord(c) - ord("a")
        reward = ale.act(actions[action_index])
        obs = ale.getScreenRGB()
        # Terminal = next char is 'R' (reset). The trace marks episode end with 'R'; we don't yield on 'R',
        # so the step we yield here is the last step of the episode iff the next character is 'R'.
        terminal = (i + 1) < len(seq) and seq[i + 1] == "R"
        yield obs, reward, terminal, action_index


def _compute_returns(rewards: List[float], terminals: List[bool], gamma: float) -> List[float]:
    """
    Compute discounted returns starting from the *next* reward (not the current one).
    G_t = r_{t+1} + gamma * r_{t+2} + ... (no r_t). At terminal[t] or when t+1 >= L, G_t = 0.
    Recurrence: returns[t] = 0 if terminal[t] or t+1 >= L else rewards[t+1] + (0 if terminal[t+1] else gamma * returns[t+1]).
    """
    L = len(rewards)
    returns = [0.0] * L
    for t in range(L - 1, -1, -1):
        if t + 1 >= L or terminals[t]:
            returns[t] = 0.0
        else:
            returns[t] = rewards[t + 1] + (
                0.0 if terminals[t + 1] else gamma * returns[t + 1]
            )
    return returns


def collect_steps_from_file(
    policy_path: str,
    max_steps: Optional[int],
    gamma: float,
) -> Tuple[List[np.ndarray], List[float], List[bool], List[int], List[float]]:
    """
    Consume from one policy file. Returns (observations, rewards, terminals, actions, returns).

    When max_steps is set: require at least max_steps + RETURNS_BUFFER_STEPS steps, collect
    that many, compute returns in reverse (terminal = no future), return first max_steps only.
    When max_steps is None: collect all steps, compute returns, return all.
    """
    obs_list: List[np.ndarray] = []
    reward_list: List[float] = []
    terminal_list: List[bool] = []
    action_list: List[int] = []

    required = (max_steps + RETURNS_BUFFER_STEPS) if max_steps is not None else None
    gen = replay_trace(policy_path)
    steps_taken = 0
    for obs, reward, terminal, action_index in gen:
        obs_list.append(obs)
        reward_list.append(reward)
        terminal_list.append(terminal)
        action_list.append(action_index)
        steps_taken += 1
        if required is not None and steps_taken >= required:
            break

    if required is not None and steps_taken < required:
        print(
            f"Error: policy file {policy_path} yielded {steps_taken} steps, "
            f"but {required} required (--steps {max_steps} + {RETURNS_BUFFER_STEPS} buffer).",
            file=sys.stderr,
        )
        sys.exit(1)

    returns_list = _compute_returns(reward_list, terminal_list, gamma)

    if max_steps is not None:
        return (
            obs_list[:max_steps],
            reward_list[:max_steps],
            terminal_list[:max_steps],
            action_list[:max_steps],
            returns_list[:max_steps],
        )
    return obs_list, reward_list, terminal_list, action_list, returns_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay policy trace JSONs and write a static dataset to .npz",
    )
    parser.add_argument(
        "policies",
        nargs="+",
        metavar="policy.json",
        help="Paths to policy trace JSON files",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        metavar="N",
        help="Max steps to collect per policy file; if set, each file must yield at least N + 5000 steps",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="Discount factor for return computation (default: 0.99)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="dataset.npz",
        help="Output .npz path (default: dataset.npz)",
    )
    args = parser.parse_args()

    all_obs: List[np.ndarray] = []
    all_rewards: List[float] = []
    all_terminals: List[bool] = []
    all_actions: List[int] = []
    all_returns: List[float] = []

    for path in args.policies:
        obs_list, reward_list, terminal_list, action_list, returns_list = collect_steps_from_file(
            path, args.steps, args.gamma
        )
        all_obs.extend(obs_list)
        all_rewards.extend(reward_list)
        all_terminals.extend(terminal_list)
        all_actions.extend(action_list)
        all_returns.extend(returns_list)

    if not all_obs:
        print("Error: no steps collected from any policy file.", file=sys.stderr)
        sys.exit(1)

    observations = np.stack(all_obs, axis=0)
    rewards = np.array(all_rewards, dtype=np.float64)
    terminals = np.array(all_terminals, dtype=bool)
    actions = np.array(all_actions, dtype=np.uint8)
    returns = np.array(all_returns, dtype=np.float64)

    np.savez_compressed(
        args.output,
        observations=observations,
        rewards=rewards,
        terminals=terminals,
        actions=actions,
        returns=returns,
    )
    T = len(all_obs)
    n_terminal = int(terminals.sum())
    terminal_indices = np.where(terminals)[0]
    print(f"Saved {T} steps to {args.output}")
    print(f"Terminals: {n_terminal} terminal steps out of {T}")
    if n_terminal > 0:
        if n_terminal <= 20:
            print(f"  Terminal indices: {terminal_indices.tolist()}")
        else:
            print(f"  First 10 terminal indices: {terminal_indices[:10].tolist()}")
            print(f"  Last 10 terminal indices: {terminal_indices[-10:].tolist()}")


if __name__ == "__main__":
    main()
