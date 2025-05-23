from __future__ import annotations

from .single_perspective import SinglePerspectiveMCTS


class ImperfectInfoMCTS(SinglePerspectiveMCTS):
    """Na\xefve determinization MCTS for games with hidden information."""

    def __init__(self, game, perspective_player, *, determinization_fn=None, **kwargs):
        super().__init__(game, perspective_player, **kwargs)
        self.determinization_fn = determinization_fn or getattr(game, "sample_determinization", None)

    def search(self, root_state, draw_callback=None):
        if self.determinization_fn is not None:
            determinized = self.determinization_fn(root_state, self.perspective_player)
        else:
            determinized = root_state
        return super().search(determinized, draw_callback)
