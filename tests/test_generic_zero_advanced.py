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

    def test_encode_state_shape(self):
        """The adapter should encode states into a 3×HxW tensor."""
        if gza is None or gza.torch is None:
            self.skipTest("PyTorch not available for encode_state test")
        game = ConnectFour()
        adapter = ConnectFourAdapter(game)
        state = game.getInitialState()
        tensor = adapter.encode_state(state, "X")
        self.assertEqual(tuple(tensor.shape), (3, game.ROWS, game.COLS))

    def test_train_step_empty_batch(self):
        """An empty batch should result in zero loss."""
        if gza is None or gza.torch is None:
            self.skipTest("PyTorch not available for train_step test")
        game = ConnectFour()
        adapter = ConnectFourAdapter(game)
        net = gza.AlphaZeroNet((3, game.ROWS, game.COLS), adapter.get_action_size())
        opt = gza.torch.optim.Adam(net.parameters(), lr=0.001)
        loss = gza.train_step(net, [], opt, "cpu", adapter)
        self.assertEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
