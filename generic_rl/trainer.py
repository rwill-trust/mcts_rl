"""Generic AlphaZero-style training utilities."""
from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy optional
    np = None

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None

from mcts.alpha_zero_mcts import AlphaZeroMCTS
from mcts.replay_buffer import PrioritizedReplayBuffer

from .adapters import GameAdapter
from .networks import AlphaZeroNet


def play_one_game(
    learning_net: AlphaZeroNet,
    game_adapter: GameAdapter,
    learning_mcts: AlphaZeroMCTS,
    temp_schedule: List[Tuple[int, float]],
    mcts_simulations: int,
    max_moves: int = 256,
    debug: bool = False,
) -> List[Tuple[dict, np.ndarray, int]]:
    """Generate self-play data for a single game."""
    if torch is None or np is None:
        raise RuntimeError("NumPy and PyTorch are required")

    state = game_adapter.getInitialState()
    history: List[Tuple[dict, np.ndarray, int]] = []
    move_no = 0
    current_temp = 1.0

    while not game_adapter.isTerminal(state) and move_no < max_moves:
        for threshold, temp_val in temp_schedule:
            if move_no < threshold:
                current_temp = temp_val
                break

        action, policy_dict = learning_mcts.get_action_policy(
            root_state=state,
            num_simulations=mcts_simulations,
            temperature=current_temp,
            debug_mcts=debug,
        )
        policy_vec = np.zeros(game_adapter.get_action_size(), dtype=np.float32)
        for a_idx, prob in policy_dict.items():
            if 0 <= a_idx < game_adapter.get_action_size():
                policy_vec[a_idx] = prob
        if abs(policy_vec.sum() - 1.0) > 1e-5 and policy_vec.sum() > 1e-5:
            policy_vec /= policy_vec.sum()
        history.append((game_adapter.copyState(state), policy_vec, 0))
        state = game_adapter.applyAction(state, action)
        move_no += 1

    winner = game_adapter.getGameOutcome(state)
    z = 0
    if winner == "X":
        z = 1
    elif winner == "O":
        z = -1

    final_history = []
    for recorded_state, policy, _ in history:
        player_at_state = game_adapter.getCurrentPlayer(recorded_state)
        value = z if player_at_state == "X" else -z
        final_history.append((recorded_state, policy, value))
    return final_history


# Simplified training step used for early refactor

def train_step(
    net: AlphaZeroNet,
    batch_experiences: list,
    opt: torch.optim.Optimizer,
    device: str,
    game_adapter: GameAdapter,
) -> float:
    if torch is None:
        raise RuntimeError("PyTorch is required for training")

    if not batch_experiences:
        return 0.0

    states = []
    policies = []
    values = []
    for s_dict, p_vec, v_val in batch_experiences:
        player = game_adapter.getCurrentPlayer(s_dict)
        states.append(game_adapter.encode_state(s_dict, player))
        policies.append(p_vec)
        values.append(v_val)

    S = torch.stack(states).to(device)
    P = torch.tensor(policies, dtype=torch.float32).to(device)
    V = torch.tensor(values, dtype=torch.float32).to(device)

    logits, v_pred = net(S)
    log_probs = torch.log_softmax(logits, dim=1)
    loss_policy = -(P * log_probs).sum(dim=1).mean()
    loss_value = torch.nn.functional.mse_loss(v_pred, V)
    loss = loss_policy + loss_value

    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.item())
