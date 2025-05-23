import random
from simple_games.three_player_card import ThreePlayerCardGame
from mcts.imperfect_info_mcts import ImperfectInfoMCTS


def main():
    game = ThreePlayerCardGame()
    full_state = game.getInitialState()
    player = "A"
    partial_state = game.getPartialState(full_state, player)

    mcts = ImperfectInfoMCTS(game, perspective_player=player, num_iterations=100)
    best_action = mcts.search(partial_state)
    print("Recommended action for", player, ":", best_action)

    state = game.applyAction(full_state, best_action)
    while not game.isTerminal(state):
        p = game.getCurrentPlayer(state)
        legal = game.getLegalActions(state)
        state = game.applyAction(state, random.choice(legal))
    print("Game outcome:", game.getGameOutcome(state))


if __name__ == "__main__":
    main()
