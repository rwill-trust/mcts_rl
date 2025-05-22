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
    is_weights: np.ndarray,
    opt: torch.optim.Optimizer,
    device: str,
    game_adapter: GameAdapter,
    augment_prob: float = 0.0,
    ent_beta_val: float = 0.0,
    debug: bool = False,
) -> tuple[float, np.ndarray]:
    """Perform one training step using KL loss and optional augmentation."""
    if torch is None or np is None:
        raise RuntimeError("PyTorch and NumPy are required for training")

    if not batch_experiences:
        return 0.0, np.array([])

    state_tensors = []
    policy_targets = []
    value_targets = []
    for s_dict, p_vec, v_val in batch_experiences:
        if random.random() < augment_prob:
            s_dict, p_vec = game_adapter.reflect_state_policy(s_dict, p_vec)
        player = game_adapter.getCurrentPlayer(s_dict)
        state_tensors.append(game_adapter.encode_state(s_dict, player))
        policy_targets.append(p_vec)
        value_targets.append(v_val)

    S = torch.stack(state_tensors).to(device)
    P_tgt = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=device)
    V_tgt = torch.tensor(value_targets, dtype=torch.float32, device=device)
    weights = torch.tensor(is_weights, dtype=torch.float32, device=device).unsqueeze(1)

    logits, v_pred = net(S)
    log_probs = torch.log_softmax(logits, dim=1)
    kl = torch.nn.functional.kl_div(log_probs, P_tgt, reduction="none").sum(dim=1)
    mse = torch.nn.functional.mse_loss(v_pred.squeeze(), V_tgt, reduction="none")

    w = weights.squeeze()
    loss_policy = (kl * w).mean()
    loss_value = (mse * w).mean()
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    loss = loss_policy + loss_value - ent_beta_val * entropy

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()

    td_errors = torch.abs(V_tgt - v_pred.squeeze()).detach().cpu().numpy()
    if debug:
        print(
            f"train_step loss {loss.item():.4f} (policy {loss_policy.item():.4f}, "
            f"value {loss_value.item():.4f}, ent {entropy.item():.4f})"
        )
    return float(loss.item()), td_errors
