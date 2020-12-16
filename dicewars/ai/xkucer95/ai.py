import logging
from .utils import *
from .expectimax_n import expectimax_n
import traceback, sys
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand


class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.players_order = players_order
        self.logger = logging.getLogger('AI')
        print('players_order', players_order)

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        turn = self.players_order.index(self.player_name)
        val, action = expectimax_n(board, len(self.players_order), turn, self.players_order)
        if action is None:
            return EndTurnCommand()
        else:
            return BattleCommand(action[0], action[1])
