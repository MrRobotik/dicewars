import logging
import torch

from os import path
from .utils import *

from .turn_simulator import TurnSimulator
from .happ_model import HoldAreaProbPredictor
from .expectimax_n import expectimax_n

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand

import traceback  # TODO: remove after debug


class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.players_order = players_order
        self.logger = logging.getLogger('AI')
        self.policy_model = HoldAreaProbPredictor()
        self.policy_model.eval()
        self.actions_buffer = []

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        try:
            return self.ai_turn_policy_only(board)
        except:
            print(traceback.format_exc())

    def ai_turn_policy_only(self, board):
        attacks = possible_attacks(board, self.player_name)
        attacks = [(s, t, p) for s, t, p in attacks if s.get_dice() >= t.get_dice()]
        attacks = sorted(attacks, key=lambda x: x[2], reverse=True)
        if len(attacks) == 0:
            return EndTurnCommand()
        source, target, _ = attacks[0]
        return BattleCommand(source.get_name(), target.get_name())
