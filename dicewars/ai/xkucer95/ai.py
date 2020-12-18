import logging
import torch

from os import path
from .utils import *
from .expectimax_n import expectimax_n
from .policy_model import PolicyModel

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand


class AI:
    def __init__(self, player_name, board, players_order, train_online=True):
        self.player_name = player_name
        self.players_order = players_order
        self.logger = logging.getLogger('AI')
        self.curr_score = None
        self.policy_model = PolicyModel(train_online)
        self.policy_model_path = 'dicewars/ai/xkucer95/models/policy_model.pt'
        if path.exists(self.policy_model_path):
            self.policy_model.load_state_dict(torch.load(self.policy_model_path))
        if train_online:
            self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=0.01)

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        if self.policy_model.train_online:
            state = state_descriptor(board, self.player_name, self.players_order)
            return self.ai_turn_policy_only(board)
        else:
            return EndTurnCommand()

    def ai_turn_policy_only(self, board):
        attacks = possible_attacks(board, self.player_name)
        attacks = [(s, t, p) for s, t, p in attacks if s.get_dice() >= t.get_dice()]
        if len(attacks) == 0:
            return EndTurnCommand()
        # data = np.stack((state_descriptor(board, self.player_name, self.players_order)), axis=0)
        source, target, _ = attacks[0]
        return BattleCommand(source.get_name(), target.get_name())

    def reward(self, reward):
        if self.policy_model.probs_buff:
            self.optimizer.zero_grad()
            self.policy_model.calc_grads(reward)
            self.optimizer.step()

    # def best_attacks(self, board, turn):
    #     attacks = possible_attacks(board, self.players_order[turn])
    #     return sorted(attacks, key=lambda x: x[2], reverse=True)[:1]
    #
    # def evaluate(self, board):
    #     val = []
    #     for player in self.players_order:
    #         val.append(max(len(region) for region in board.get_players_regions(player)))
    #     return np.asarray(val)
