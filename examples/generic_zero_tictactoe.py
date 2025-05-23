#!/usr/bin/env python3
"""Example entry point for the generic AlphaZero trainer with TicTacToe."""
from __future__ import annotations

import argparse

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None

from simple_games.tic_tac_toe import TicTacToe
from mcts.alpha_zero_mcts import AlphaZeroMCTS

from generic_rl.adapters import TicTacToeAdapter
from generic_rl.networks import AlphaZeroNet
from generic_rl.trainer import play_one_game, train_step


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generic AlphaZero TicTacToe example")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--games-per-epoch", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    return p


def run(cli_args=None) -> None:
    args = parser().parse_args([] if cli_args is None else cli_args)
    if torch is None:
        raise RuntimeError("PyTorch is required to run the generic trainer")

    game = TicTacToe()
    adapter = TicTacToeAdapter(game)
    net = AlphaZeroNet((3, 3, 3), adapter.get_action_size())
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

    def model_fn(batch):
        net.eval()
        logits, value = net(batch)
        net.train()
        return logits, value.unsqueeze(-1)

    mcts = AlphaZeroMCTS(adapter, model_fn, torch.device("cpu"), c_puct=1.4)

    buffer = []
    temp_schedule = [(5, 1.0), (float('inf'), 0.5)]

    for _ in range(args.epochs):
        for _ in range(args.games_per_epoch):
            data = play_one_game(net, adapter, mcts, temp_schedule, mcts_simulations=5)
            buffer.extend(data)
        if buffer:
            batch = buffer[: args.batch_size]
            loss = train_step(net, batch, opt, "cpu", adapter)
            print(f"train_step loss: {loss:.4f}")


if __name__ == "__main__":
    run()
