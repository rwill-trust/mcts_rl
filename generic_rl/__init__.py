"""Generic reinforcement learning utilities."""

from .adapters import GameAdapter, ConnectFourAdapter, TicTacToeAdapter
from .networks import AlphaZeroNet

# Avoid importing heavy trainer dependencies unless explicitly requested.
def play_one_game(*args, **kwargs):  # pragma: no cover - thin wrapper
    from .trainer import play_one_game as _impl
    return _impl(*args, **kwargs)


def train_step(*args, **kwargs):  # pragma: no cover - thin wrapper
    from .trainer import train_step as _impl
    return _impl(*args, **kwargs)

__all__ = [
    "GameAdapter",
    "ConnectFourAdapter",
    "TicTacToeAdapter",
    "AlphaZeroNet",
    "play_one_game",
    "train_step",
]
