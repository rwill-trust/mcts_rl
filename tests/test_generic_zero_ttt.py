import unittest

from generic_rl.adapters import TicTacToeAdapter
from simple_games.tic_tac_toe import TicTacToe

try:
    import numpy  # noqa: F401
    from examples import generic_zero_tictactoe as gzt
except Exception:
    gzt = None
try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover - torch optional
    torch = None


class TestGenericZeroTicTacToe(unittest.TestCase):
    def test_parser_defaults(self):
        if gzt is None:
            self.skipTest("numpy or example script not available")
        p = gzt.parser()
        args = p.parse_args([])
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.games_per_epoch, 1)
        self.assertEqual(args.batch_size, 4)

    def test_adapter_shapes(self):
        game = TicTacToe()
        adapter = TicTacToeAdapter(game)
        self.assertEqual(adapter.get_action_size(), 9)
        if torch is None:
            self.skipTest("PyTorch not available")
        state = game.getInitialState()
        encoded = adapter.encode_state(state, "X")
        self.assertEqual(tuple(encoded.shape), (3, 3, 3))


if __name__ == "__main__":
    unittest.main()
