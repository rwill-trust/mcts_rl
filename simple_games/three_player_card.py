import random

class ThreePlayerCardGame:
    """Simple 3-player card swapping game with hidden information."""

    PLAYERS = ["A", "B", "C"]
    DECK_CARDS = [1, 2, 3, 4, 5]

    def getInitialState(self):
        deck = self.DECK_CARDS[:]
        random.shuffle(deck)
        cards = {p: deck.pop() for p in self.PLAYERS}
        return {"deck": deck, "cards": cards, "current_index": 0}

    def copyState(self, state):
        return {
            "deck": state["deck"][:],
            "cards": state["cards"].copy(),
            "current_index": state["current_index"],
        }

    def getCurrentPlayer(self, state):
        return self.PLAYERS[state["current_index"]]

    def getLegalActions(self, state):
        if state["current_index"] >= len(self.PLAYERS):
            return []
        return ["keep", "swap"]

    def applyAction(self, state, action):
        new_state = self.copyState(state)
        if state["current_index"] >= len(self.PLAYERS):
            return new_state
        player = self.getCurrentPlayer(state)
        if action == "swap" and new_state["deck"]:
            new_card = random.choice(new_state["deck"])
            new_state["deck"].remove(new_card)
            new_state["deck"].append(new_state["cards"][player])
            new_state["cards"][player] = new_card
        new_state["current_index"] += 1
        return new_state

    def getGameOutcome(self, state):
        if state["current_index"] < len(self.PLAYERS):
            return None
        cards = state["cards"]
        best = max(cards.values())
        winners = [p for p, c in cards.items() if c == best]
        if len(winners) == 1:
            return winners[0]
        return "Draw"

    def isTerminal(self, state):
        return self.getGameOutcome(state) is not None

    def getPartialState(self, state, player):
        partial = self.copyState(state)
        for p in self.PLAYERS:
            if p != player:
                partial["cards"][p] = None
        return partial

    def sample_determinization(self, partial_state, player):
        known = partial_state["cards"][player]
        deck_pool = self.DECK_CARDS[:]
        deck_pool.remove(known)
        random.shuffle(deck_pool)
        cards = {player: known}
        for p in self.PLAYERS:
            if p == player:
                continue
            cards[p] = deck_pool.pop()
        deck = deck_pool
        return {"deck": deck, "cards": cards, "current_index": partial_state["current_index"]}

    def simulateRandomPlayout(self, state, perspectivePlayer, max_depth=10, eval_func=None, weights=None):
        temp = self.copyState(state)
        depth = 0
        while not self.isTerminal(temp) and depth < max_depth:
            player = self.getCurrentPlayer(temp)
            legal = self.getLegalActions(temp)
            action = random.choice(legal)
            temp = self.applyAction(temp, action)
            depth += 1
        outcome = self.getGameOutcome(temp)
        if outcome == perspectivePlayer:
            return 1.0
        elif outcome == "Draw" or outcome is None:
            return 0.0
        else:
            return -1.0

    def evaluateState(self, perspectivePlayer, state, weights=None):
        outcome = self.getGameOutcome(state)
        if outcome == perspectivePlayer:
            return 1.0
        elif outcome == "Draw" or outcome is None:
            return 0.0
        else:
            return -1.0
