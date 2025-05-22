from __future__ import annotations
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy optional
    np = None

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None


class GameAdapter:
    """Abstract adapter exposing the game API required by the generic trainer."""

    def getInitialState(self) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def getCurrentPlayer(self, state: Any) -> str:
        raise NotImplementedError

    def getLegalActions(self, state: Any) -> List[int]:
        raise NotImplementedError

    def applyAction(self, state: Any, action: int) -> Any:
        raise NotImplementedError

    def isTerminal(self, state: Any) -> bool:
        raise NotImplementedError

    def getGameOutcome(self, state: Any) -> str | None:
        raise NotImplementedError

    def encode_state(self, state: Any, perspective: str) -> 'torch.Tensor':
        raise NotImplementedError

    def copyState(self, state: Any) -> Any:
        raise NotImplementedError

    def get_action_size(self) -> int:
        raise NotImplementedError

    def reflect_state_policy(self, state: Any, policy: np.ndarray) -> Tuple[Any, np.ndarray]:  # pragma: no cover - optional augmentation
        """Return a horizontally reflected version of ``state`` and ``policy``.

        Games that support data augmentation should override this method. The
        default implementation returns the inputs unchanged.
        """
        return state, policy


# Example implementation for ConnectFour
from simple_games.connect_four import ConnectFour

BOARD_H = ConnectFour.ROWS
BOARD_W = ConnectFour.COLS

class ConnectFourAdapter(GameAdapter):
    def __init__(self, c4_game: ConnectFour):
        self.c4_game = c4_game
        self.action_size = self.c4_game.get_action_size()

    def getInitialState(self) -> Dict:
        return self.c4_game.getInitialState()

    def getCurrentPlayer(self, state: Dict) -> str:
        return self.c4_game.getCurrentPlayer(state)

    def getLegalActions(self, state: Dict) -> List[int]:
        return self.c4_game.getLegalActions(state)

    def applyAction(self, state: Dict, action_int: int) -> Dict:
        return self.c4_game.applyAction(state, action_int)

    def isTerminal(self, state: Dict) -> bool:
        return self.c4_game.isTerminal(state)

    def getGameOutcome(self, state: Dict) -> str | None:
        return self.c4_game.getGameOutcome(state)

    def encode_state(self, state: Dict, player_perspective: str) -> 'torch.Tensor':
        """Return a 3×HxW tensor representing ``state`` from ``player_perspective``."""
        if torch is None:
            raise RuntimeError("PyTorch is required for encoding state")
        t = torch.zeros(3, BOARD_H, BOARD_W)
        for r in range(BOARD_H):
            for c in range(BOARD_W):
                piece = state["board"][r][c]
                if piece == player_perspective:
                    t[0, r, c] = 1.0
                elif piece is not None:
                    t[1, r, c] = 1.0
        if state["current_player"] == player_perspective:
            t[2].fill_(1.0)
        return t

    def copyState(self, state: Dict) -> Dict:
        return self.c4_game.copyState(state)

    def get_action_size(self) -> int:
        return self.action_size

    def reflect_state_policy(self, state: Dict, policy: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """Return a horizontally reflected version of the board and policy."""
        if np is None:
            raise RuntimeError("NumPy is required for reflection")
        reflected = {"board": [row[::-1] for row in state["board"]], "current_player": state["current_player"]}
        return reflected, policy[::-1].copy()
