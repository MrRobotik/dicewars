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
    def __init__(self, player_name, board, players_order, on_policy=False):
        self.player_name = player_name
        self.players_order = players_order
        self.logger = logging.getLogger('AI')
        self.policy_model = PolicyModel(on_policy)
        self.policy_model_path = 'dicewars/ai/xkucer95/models/policy_model.pt'
        if path.exists(self.policy_model_path):
            self.policy_model.load_state_dict(torch.load(self.policy_model_path))
        if on_policy:
            self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=1.e-3)
            self.batch_size = 128

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        try:
            return self.ai_turn_on_policy(board)
        except:
            print(traceback.format_exc())

    def ai_turn_on_policy(self, board):
        attacks = possible_attacks(board, self.player_name)
        attacks = [(s, t, p) for s, t, p in attacks if s.get_dice() >= t.get_dice()]
        if len(attacks) == 0:
            return EndTurnCommand()
        data_in = []
        x_curr = state_descriptor(board, self.player_name, self.players_order)
        for source, target, succ_prob in attacks:
            ts = TurnSimulator(board)
            ts.do_attack(source, target, succ_prob, True)
            if board.nb_players_alive() == 1:
                return BattleCommand(source.get_name(), target.get_name())
            x_next = state_descriptor(board, self.player_name, self.players_order)
            x = np.concatenate((np.asarray([succ_prob], dtype=np.float32), x_next / x_curr))
            ts.undo_attack()
            data_in.append(x)

        data_in = np.vstack(data_in)
        action = self.policy_model.select_action(data_in)
        if action is None:
            return EndTurnCommand()
        source, target, _ = attacks[action]
        return BattleCommand(source.get_name(), target.get_name())

    def give_reward(self, reward, game_end=False):
        if self.policy_model.on_policy:
            self.policy_model.give_reward(reward)
            if len(self.policy_model.train_buff) >= self.batch_size or game_end:
                self.optimizer.zero_grad()
                self.policy_model.backward()
                self.optimizer.step()
