import logging

from .utils import *

from .turn_simulator import TurnSimulator
from .happ_model import HoldAreaProbPredictor
from .wpp_model import WinProbPredictor
from .expectimax_n import expectimax_n, Heuristics

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand


class AI:
    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.players_order = players_order
        self.logger = logging.getLogger('AI')
        self.happ_model = HoldAreaProbPredictor()
        self.happ_model.eval()
        self.wpp_model = WinProbPredictor()
        self.wpp_model.eval()
        self.heuristics = Heuristics(self.eval_attacks, self.eval_game, players_order)

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        if time_left < 2.0:
            print('fallback')
            return self.ai_turn_impl_2(board)
        depth = 4
        return self.ai_turn_impl_3(board, depth)

    def ai_turn_impl_1(self, board):
        attacks = possible_attacks(board, self.player_name)
        attacks = [(s, t, p) for s, t, p in attacks if s.get_dice() >= t.get_dice()]
        if len(attacks) == 0:
            return EndTurnCommand()
        attacks = sorted(attacks, key=lambda x: x[2], reverse=True)
        source, target, _ = attacks[0]
        return BattleCommand(source.get_name(), target.get_name())

    def ai_turn_impl_2(self, board):
        attacks = possible_attacks(board, self.player_name)
        attacks = [(s, t, p) for s, t, p in attacks if s.get_dice() >= t.get_dice()]
        if len(attacks) == 0:
            return EndTurnCommand()
        probs = self.eval_attacks(board, attacks) * np.asarray([p for _, _, p in attacks])
        best = int(np.argmax(probs))
        if probs[best] > 0.1:
            source, target, _ = attacks[best]
            return BattleCommand(source.get_name(), target.get_name())
        return EndTurnCommand()

    def ai_turn_impl_3(self, board, depth=1):
        turn = self.players_order.index(self.player_name)
        n = len(self.players_order)
        _, act = expectimax_n(board, n, turn, depth, self.heuristics)
        if act is None:
            return EndTurnCommand()
        source, target = act
        return BattleCommand(source.get_name(), target.get_name())

    def eval_attacks(self, board, attacks):
        x_source = []
        x_target = []
        for source, target, succ_prob in attacks:
            ts = TurnSimulator(board)
            ts.do_attack(source, target, succ_prob, True)
            x_source.append(area_descriptor(source, board))
            x_target.append(area_descriptor(target, board))
            ts.undo_attack()

        with torch.no_grad():
            p_source = self.happ_model(torch.from_numpy(np.vstack(x_source))).detach().numpy().ravel()
            p_target = self.happ_model(torch.from_numpy(np.vstack(x_target))).detach().numpy().ravel()
        return p_source * p_target

    def eval_game(self, board, player_name):
        x = game_descriptor(board, player_name, self.players_order)
        return self.wpp_model(torch.from_numpy(x)).detach().numpy().squeeze()
