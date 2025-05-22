import unittest

from generic_rl.adapters import ConnectFourAdapter
from simple_games.connect_four import ConnectFour

try:
    import numpy as np  # noqa: F401
    from examples import generic_zero_advanced as gza
except Exception:
    gza = None


class TestGenericZeroAdvanced(unittest.TestCase):
    def test_parser_defaults(self):
        if gza is None:
            self.skipTest("numpy or example script not available")
        p = gza.parser()
        args = p.parse_args([])
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.games_per_epoch, 1)
        self.assertEqual(args.batch_size, 4)

    def test_adapter_action_size(self):
        game = ConnectFour()
        adapter = ConnectFourAdapter(game)
        self.assertEqual(adapter.get_action_size(), ConnectFour.COLS)

    def test_reflection(self):
        try:
            import numpy as np
        except Exception:
            self.skipTest("numpy not available")
        game = ConnectFour()
        adapter = ConnectFourAdapter(game)
        state = game.getInitialState()
        policy = np.arange(adapter.get_action_size(), dtype=np.float32)
        refl_state, refl_pol = adapter.reflect_state_policy(state, policy)
        self.assertEqual(refl_state["board"], [row[::-1] for row in state["board"]])
        self.assertTrue(np.array_equal(refl_pol, policy[::-1]))


if __name__ == "__main__":
    unittest.main()
