import logging
import torch

from os import path
from .utils import *

from .turn_simulator import TurnSimulator
from .policy_model import PolicyModel
from .expectimax_n import expectimax_n

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand

import traceback  # TODO: remove after debug


class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.players_order = players_order
        self.logger = logging.getLogger('AI')
        self.policy_model = PolicyModel()
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
        print(attacks)
        if len(attacks) == 0:
            return EndTurnCommand()
        # data_in = []
        # x_curr = state_descriptor(board, self.player_name, self.players_order)
        # for source, target, succ_prob in attacks:
        #     ts = TurnSimulator(board)
        #     ts.do_attack(source, target, succ_prob, True)
        #     if board.nb_players_alive() == 1:
        #         return BattleCommand(source.get_name(), target.get_name())
        #     x_next = state_descriptor(board, self.player_name, self.players_order)
        #     survival_prob_1 = survival_prob(board, source, self.player_name)
        #     survival_prob_2 = survival_prob(board, target, self.player_name)
        #     attack_specific = [succ_prob, survival_prob_1, survival_prob_2]
        #     x = np.concatenate((np.asarray(attack_specific, dtype=np.float32), x_curr, x_next))
        #     ts.undo_attack()
        #     data_in.append(x)
        #
        # data_in = np.vstack(data_in)
        # action = self.policy_model.select_action(data_in, False)
        # if action is None:
        #     return EndTurnCommand()
        # self.actions_buffer.append(data_in[action])
        source, target, _ = attacks[0]
        return BattleCommand(source.get_name(), target.get_name())

    def give_reward(self, reward):
        discount = 0.95
        n = 0
        with open('dicewars/ai/xkucer95/models/policy_model_trn.dat', 'a') as f:
            while self.actions_buffer:
                x = self.actions_buffer.pop()
                r = discount**n * reward
                f.write('{} {}\n'.format(' '.join([str(v) for v in x]), r))
                n += 1
