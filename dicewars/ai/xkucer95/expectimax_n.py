from .utils import *
from .turn_simulator import TurnSimulator
import numpy as np


class Heuristics:
    def __init__(self, eval_attacks_fn, eval_game_fn, players_order, root_player):
        self.eval_attacks_fn = eval_attacks_fn
        self.eval_game_fn = eval_game_fn
        self.players_order = players_order
        self.root_player = root_player

    def is_root_player_turn(self, turn: int):
        return self.players_order[turn] == self.root_player

    def get_best_attacks(self, board: Board, turn: int):
        attacks = possible_attacks(board, self.players_order[turn])
        attacks = [(s, t, p) for s, t, p in attacks if s.get_dice() >= t.get_dice()]
        if len(attacks) > 0:
            attacks = sorted(attacks, key=lambda x: x[2], reverse=True)[:10]
            probs = self.eval_attacks_fn(board, attacks) * np.asarray([p for _, _, p in attacks])
            indices = [i for i in np.argsort(-probs) if probs[i] > 0.10]
            if self.is_root_player_turn(turn):
                for i in indices:
                    if (probs[i] / probs[indices[0]]) > 0.95:
                        yield attacks[i]
            elif len(indices) > 0:
                yield attacks[indices[0]]

    def evaluate(self, board: Board):
        val = []
        for player_name in self.players_order:
            areas = board.get_player_areas(player_name)
            if   len(areas) == len(board.areas):
                val.append(1.)
            elif len(areas) == 0:
                val.append(0.)
            else:
                val.append(float(self.eval_game_fn(board, player_name)))
        return np.asarray(val)


def expectimax_n(board: Board, depth: int, turn: int, n: int, heuristics: Heuristics):
    if depth == 0 or board.nb_players_alive() == 1:
        return heuristics.evaluate(board), None

    next_turn = (turn + 1) % n if (depth - 1) % 2 == 0 else turn
    best_val = np.full(n, -np.infty)
    best_act = None

    if depth > 7:
        next_turn = turn

    attacks = tuple(heuristics.get_best_attacks(board, turn))
    if depth == 10 and len(attacks) < 2:
        if len(attacks) > 0:
            source, target, _ = attacks[0]
            best_act = source, target
        return None, best_act

    for source, target, succ_prob in attacks:
        val = expand_chances(source, target, succ_prob,
                             board, depth - 1,
                             next_turn, n, heuristics)
        if val[turn] > best_val[turn]:
            best_val = val
            best_act = source, target

    if best_val[turn] < 0.90 and np.isneginf(best_val[turn]):
        val, _ = expectimax_n(board, depth - 1, next_turn, n, heuristics)
        if val[turn] > best_val[turn]:
            best_val = val
            best_act = None

    return best_val, best_act


def expand_chances(source: Area, target: Area, succ_prob: float, board: Board, *args):
    ts = TurnSimulator(board)
    ts.do_attack(source, target, succ_prob, True)
    p_succ = ts.curr_prob
    v_succ, _ = expectimax_n(board, *args)
    ts.undo_attack()
    ts.do_attack(source, target, succ_prob, False)
    p_fail = ts.curr_prob
    v_fail, _ = expectimax_n(board, *args)
    ts.undo_attack()
    return (p_succ * v_succ) + (p_fail * v_fail)
