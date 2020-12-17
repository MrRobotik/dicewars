import logging
import numpy as np
from .utils import *

from .expectimax_n import expectimax_n
from .policy_model import PolicyModel

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand


class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.players_order = players_order
        self.logger = logging.getLogger('AI')
        self.policy_model = PolicyModel()

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        for source, target, prob in possible_attacks(board, self.player_name):
            if source.get_dice() < target.get_dice():
                continue


        return EndTurnCommand()

    # def best_attacks(self, board, turn):
    #     attacks = possible_attacks(board, self.players_order[turn])
    #     return sorted(attacks, key=lambda x: x[2], reverse=True)[:1]
    #
    # def evaluate(self, board):
    #     val = []
    #     for player in self.players_order:
    #         val.append(max(len(region) for region in board.get_players_regions(player)))
    #     return np.asarray(val)
