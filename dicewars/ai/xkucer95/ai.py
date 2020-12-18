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
        self.train_online = train_online
        self.policy_model = PolicyModel()
        self.policy_model_path = 'dicewars/ai/xkucer95/models/policy_model.pt'
        if path.exists(self.policy_model_path):
            self.policy_model.load_state_dict(torch.load(self.policy_model_path))
        # self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=0.01)
        # torch.save(self.policy_model.state_dict(), self.policy_model_path)
        print(players_order)
        print('player_name', player_name)

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        if self.train_online:
            try:
                return self.ai_turn_policy_only(board)
            except Exception as e:
                print(e)
        else:
            return EndTurnCommand()

    def ai_turn_policy_only(self, board):
        attacks = possible_attacks(board, self.player_name)
        attacks = list(filter(lambda x: x[0].get_dice() >= x[1].get_dice(), attacks))
        if len(attacks) == 0:
            return EndTurnCommand()
        data = np.stack((make_attack_descriptor(board, s, t) for s, t, _ in attacks), axis=0)
        data[:, 2:] = standardize_data(data[:, 2:], axis=0)
        action = self.policy_model.select_action(data.astype(np.float32))
        source, target, _ = attacks[action]
        return BattleCommand(source.get_name(), target.get_name())

    def reward(self, r):
        print('reward', r)

    # def best_attacks(self, board, turn):
    #     attacks = possible_attacks(board, self.players_order[turn])
    #     return sorted(attacks, key=lambda x: x[2], reverse=True)[:1]
    #
    # def evaluate(self, board):
    #     val = []
    #     for player in self.players_order:
    #         val.append(max(len(region) for region in board.get_players_regions(player)))
    #     return np.asarray(val)
