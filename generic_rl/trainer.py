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


def train_step(
    net: AlphaZeroNet,
    batch_experiences: list,
    opt: "torch.optim.Optimizer",
    device: str,
    game_adapter: GameAdapter,
    is_weights: np.ndarray | None = None,
    augment_prob: float = 0.0,
    ent_beta_val: float = 0.01,
    debug: bool = False,
) -> Tuple[float, np.ndarray]:
    """Train ``net`` on one batch of experiences.

    Parameters
    ----------
    net : AlphaZeroNet
        The neural network being optimised.
    batch_experiences : list
        Each item is ``(state_dict, policy_vector, value)``.
    opt : torch.optim.Optimizer
        Optimiser for ``net``.
    device : str
        Device string for tensors (e.g. ``"cpu"``).
    game_adapter : GameAdapter
        Adapter providing encoding and augmentation hooks.
    is_weights : np.ndarray | None
        Importance-sampling weights from a replay buffer.
    augment_prob : float
        Probability of applying ``reflect_state_policy`` to each sample.
    ent_beta_val : float
        Entropy regularisation coefficient.
    debug : bool, optional
        If ``True`` print diagnostic information.
    """

    if torch is None or np is None:
        raise RuntimeError("PyTorch and NumPy are required for training")

    if not batch_experiences:
        if debug:
            print("[train_step] empty batch")
        return 0.0, np.array([])

    if is_weights is None:
        is_weights = np.ones(len(batch_experiences), dtype=np.float32)

    states, policies, values = [], [], []
    for s_dict, p_vec, v_val in batch_experiences:
        s_use, p_use = s_dict, p_vec
        if random.random() < augment_prob:
            s_use, p_use = game_adapter.reflect_state_policy(s_dict, p_vec)
        player = game_adapter.getCurrentPlayer(s_use)
        states.append(game_adapter.encode_state(s_use, player))
        policies.append(p_use)
        values.append(v_val)

    S = torch.stack(states).to(device)
    P_tgt = torch.tensor(np.array(policies), dtype=torch.float32, device=device)
    V_tgt = torch.tensor(values, dtype=torch.float32, device=device)
    w = torch.tensor(is_weights, dtype=torch.float32, device=device)

    logits, V_pred = net(S)
    logP_pred = torch.log_softmax(logits, dim=1)

    loss_policy = (torch.nn.functional.kl_div(logP_pred, P_tgt, reduction="none").sum(dim=1) * w).mean()
    loss_value = (torch.nn.functional.mse_loss(V_pred.squeeze(), V_tgt, reduction="none") * w).mean()

    entropy = -(torch.exp(logP_pred) * logP_pred).sum(dim=1).mean()
    total_loss = loss_policy + loss_value - ent_beta_val * entropy

    if debug:
        print(f"loss_p={loss_policy.item():.4f} loss_v={loss_value.item():.4f} ent={entropy.item():.4f}")

    opt.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()

    td_errors = np.abs(V_tgt.detach().cpu().numpy() - V_pred.squeeze().detach().cpu().numpy())
    return float(total_loss.item()), td_errors
