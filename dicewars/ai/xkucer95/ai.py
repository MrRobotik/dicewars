import logging
import torch.optim

from .utils import *
from .expectimax_n import expectimax_n
from .policy_model import PolicyModel

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand

import traceback, sys
class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.players_order = players_order
        self.logger = logging.getLogger('AI')
        self.policy_model = PolicyModel()
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=0.01)

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        attacks = []
        for source, target, prob in possible_attacks(board, self.player_name):
            if source.get_dice() < target.get_dice():
                continue
            attacks.append((source, target, prob))
        if len(attacks) == 0:
            return EndTurnCommand()
        x = np.stack((make_attack_descriptor(board, s, t) for s, t, _ in attacks), axis=0)
        x[:, 2:] = standardize_data(x[:, 2:], axis=0)
        action = self.policy_model.choose_action(x.astype(np.float32))
        source, target, _ = attacks[action]
        return BattleCommand(source.get_name(), target.get_name())

    # def best_attacks(self, board, turn):
    #     attacks = possible_attacks(board, self.players_order[turn])
    #     return sorted(attacks, key=lambda x: x[2], reverse=True)[:1]
    #
    # def evaluate(self, board):
    #     val = []
    #     for player in self.players_order:
    #         val.append(max(len(region) for region in board.get_players_regions(player)))
    #     return np.asarray(val)
