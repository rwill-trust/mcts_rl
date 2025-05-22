#!/usr/bin/env python3
"""Example entry point for the generic AlphaZero trainer.

This script demonstrates how to wire up the generic training utilities
with the Connect Four game.  It intentionally keeps the main loop very
simple so the refactoring can evolve iteratively."""
from __future__ import annotations

import argparse

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy optional
    np = None

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None

from simple_games.connect_four import ConnectFour
from mcts.alpha_zero_mcts import AlphaZeroMCTS

from generic_rl.adapters import ConnectFourAdapter
from generic_rl.networks import AlphaZeroNet
from generic_rl.trainer import play_one_game, train_step


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generic AlphaZero training example")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--games-per-epoch", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    return p


def run(cli_args=None) -> None:
    args = parser().parse_args([] if cli_args is None else cli_args)
    if torch is None:
        raise RuntimeError("PyTorch is required to run the generic trainer")

    game = ConnectFour()
    adapter = ConnectFourAdapter(game)
    net = AlphaZeroNet((3, game.ROWS, game.COLS), adapter.get_action_size())
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

    def model_fn(batch):
        net.eval()
        logits, value = net(batch)
        net.train()
        return logits, value.unsqueeze(-1)

    mcts = AlphaZeroMCTS(adapter, model_fn, torch.device("cpu"), c_puct=1.4)

    buffer = []
    temp_schedule = [(10, 1.0), (float('inf'), 0.5)]

    for _ in range(args.epochs):
        for _ in range(args.games_per_epoch):
            data = play_one_game(net, adapter, mcts, temp_schedule, mcts_simulations=10)
            buffer.extend(data)
        if buffer:
            batch = buffer[: args.batch_size]
            if np is None:
                raise RuntimeError("NumPy is required for training")
            is_w = np.ones(len(batch), dtype=np.float32)
            loss, _ = train_step(
                net,
                batch,
                is_w,
                opt,
                "cpu",
                adapter,
                augment_prob=0.5,
                ent_beta_val=0.01,
            )
            print(f"train_step loss: {loss:.4f}")


if __name__ == "__main__":
    run()
