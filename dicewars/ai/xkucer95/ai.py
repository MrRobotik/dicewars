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
        self.happ_model = HoldAreaProbPredictor()
        self.happ_model.eval()
        self.actions_buffer = []

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        try:
            return self.ai_turn_v2(board)
            # return self.ai_turn_v1(board)
        except:
            print(traceback.format_exc())

    def ai_turn_v1(self, board):
        attacks = possible_attacks(board, self.player_name)
        attacks = [(s, t, p) for s, t, p in attacks if s.get_dice() >= t.get_dice()]
        if len(attacks) == 0:
            return EndTurnCommand()
        attacks = sorted(attacks, key=lambda x: x[2], reverse=True)
        source, target, _ = attacks[0]
        return BattleCommand(source.get_name(), target.get_name())

    def ai_turn_v2(self, board):
        attacks = possible_attacks(board, self.player_name)
        attacks = [(s, t, p) for s, t, p in attacks if s.get_dice() >= t.get_dice()]
        if len(attacks) == 0:
            return EndTurnCommand()

        x_source = []
        x_target = []
        succ_probs = []
        for source, target, succ_prob in attacks:
            ts = TurnSimulator(board)
            ts.do_attack(source, target, succ_prob, True)
            x_source.append(area_descriptor(source, board))
            x_target.append(area_descriptor(target, board))
            succ_probs.append(succ_prob)
            ts.undo_attack()

        with torch.no_grad():
            p_source = self.happ_model(torch.from_numpy(np.vstack(x_source))).detach().numpy().ravel()
            p_target = self.happ_model(torch.from_numpy(np.vstack(x_target))).detach().numpy().ravel()

        final_probs = p_source * p_target * np.asarray(succ_probs)
        best = int(np.argmax(final_probs))
        if final_probs[best] < 0.10:
            return EndTurnCommand()
        source, target, _ = attacks[best]
        return BattleCommand(source.get_name(), target.get_name())
