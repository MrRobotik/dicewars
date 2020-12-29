from .utils import *
from .turn_simulator import TurnSimulator
import numpy as np


class Heuristics:
    def __init__(self, eval_attacks_fn, eval_game_fn, players_order):
        self.eval_attacks_fn = eval_attacks_fn
        self.eval_game_fn = eval_game_fn
        self.players_order = players_order

    def get_best_attacks(self, board: Board, turn: int):
        attacks = possible_attacks(board, self.players_order[turn])
        attacks = [(s, t, p) for s, t, p in attacks if s.get_dice() >= t.get_dice()]
        if len(attacks) > 0:
            attacks = sorted(attacks, key=lambda x: x[2], reverse=True)[:20]
            probs = self.eval_attacks_fn(board, attacks) * np.asarray([p for _, _, p in attacks])
            indices = [i for i in np.argsort(-probs) if probs[i] > 0.1]
            for i in indices:
                yield attacks[i]

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

    next_turn = (turn + 1) % n
    best_val = np.full(n, -np.infty)
    best_act = None
    attacks = heuristics.get_best_attacks(board, turn)

    for source, target, succ_prob in attacks:
        val = expand_chances(source, target, succ_prob,
                             board, depth - 1,
                             next_turn, n, heuristics)
        if val[turn] > best_val[turn]:
            best_val = val
            best_act = source, target

    if best_val[turn] < 0.9 and np.isneginf(best_val[turn]):
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
