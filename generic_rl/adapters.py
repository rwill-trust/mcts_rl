from __future__ import annotations
from typing import Any, Dict, List

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


# Example implementation for TicTacToe
from simple_games.tic_tac_toe import TicTacToe

TTT_SIZE = 3


class TicTacToeAdapter(GameAdapter):
    """Adapter wiring the :class:`TicTacToe` game to the generic trainer."""

    def __init__(self, ttt_game: TicTacToe):
        self.ttt_game = ttt_game
        self.action_size = self.ttt_game.get_action_size()
        # Precompute mapping between action indices and board coordinates
        self._index_to_action = [
            (r, c) for r in range(TTT_SIZE) for c in range(TTT_SIZE)
        ]
        self._action_to_index = {
            action: idx for idx, action in enumerate(self._index_to_action)
        }

    def getInitialState(self) -> Dict:
        return self.ttt_game.getInitialState()

    def getCurrentPlayer(self, state: Dict) -> str:
        return self.ttt_game.getCurrentPlayer(state)

    def getLegalActions(self, state: Dict) -> List[int]:
        actions = self.ttt_game.getLegalActions(state)
        return [self._action_to_index[a] for a in actions]

    def applyAction(self, state: Dict, action_int: int) -> Dict:
        coord = self._index_to_action[action_int]
        return self.ttt_game.applyAction(state, coord)

    def isTerminal(self, state: Dict) -> bool:
        return self.ttt_game.isTerminal(state)

    def getGameOutcome(self, state: Dict) -> str | None:
        return self.ttt_game.getGameOutcome(state)

    def encode_state(self, state: Dict, player_perspective: str) -> 'torch.Tensor':
        """Return a 3×3×3 tensor encoding ``state`` from ``player_perspective``."""
        if torch is None:
            raise RuntimeError("PyTorch is required for encoding state")
        t = torch.zeros(3, TTT_SIZE, TTT_SIZE)
        for r in range(TTT_SIZE):
            for c in range(TTT_SIZE):
                piece = state["board"][r][c]
                if piece == player_perspective:
                    t[0, r, c] = 1.0
                elif piece is not None:
                    t[1, r, c] = 1.0
        if state["current_player"] == player_perspective:
            t[2].fill_(1.0)
        return t

    def copyState(self, state: Dict) -> Dict:
        return self.ttt_game.copyState(state)

    def get_action_size(self) -> int:
        return self.action_size
