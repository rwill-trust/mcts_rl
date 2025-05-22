import unittest

from generic_rl.adapters import ConnectFourAdapter
from simple_games.connect_four import ConnectFour

try:
    import numpy  # noqa: F401
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


if __name__ == "__main__":
    unittest.main()
