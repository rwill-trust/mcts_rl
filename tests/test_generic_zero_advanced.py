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

    def test_state_copy_independence(self):
        """Ensure copyState returns a deep copy so modifications do not affect the original."""
        game = ConnectFour()
        adapter = ConnectFourAdapter(game)
        state = adapter.getInitialState()
        copied = adapter.copyState(state)
        copied["board"][0][0] = "X"
        self.assertIsNone(state["board"][0][0])

    def test_encode_state_channels(self):
        """Check that encode_state marks player's pieces, opponent pieces and player turn."""
        if gza is None or gza.torch is None:
            self.skipTest("torch or numpy not available")
        game = ConnectFour()
        adapter = ConnectFourAdapter(game)
        state = adapter.getInitialState()
        state["board"][0][0] = "X"
        encoded = adapter.encode_state(state, "X")
        self.assertEqual(list(encoded.shape), [3, game.ROWS, game.COLS])
        self.assertEqual(encoded[0, 0, 0].item(), 1.0)
        self.assertEqual(float(encoded[1].sum().item()), 0.0)
        self.assertEqual(float(encoded[2].mean().item()), 1.0)

    def test_train_step_empty_batch(self):
        """train_step should return zero loss when given an empty batch."""
        if gza is None or gza.torch is None:
            self.skipTest("torch or numpy not available")
        game = ConnectFour()
        adapter = ConnectFourAdapter(game)
        net = gza.AlphaZeroNet((3, game.ROWS, game.COLS), adapter.get_action_size())
        opt = gza.torch.optim.SGD(net.parameters(), lr=0.001)
        loss, td = gza.train_step(net, [], opt, "cpu", adapter)
        self.assertEqual(loss, 0.0)
        self.assertEqual(len(td), 0)


if __name__ == "__main__":
    unittest.main()
