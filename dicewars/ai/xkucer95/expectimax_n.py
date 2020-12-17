from .utils import *
import numpy as np


def expectimax_n(board, depth, turn, n, heuristics):
    if board.nb_players_alive() == 1 or depth == 0:
        return heuristics.evaluate(board), None
    next_turn = (turn + 1) % n
    best_val = np.full(n, -np.infty)
    best_act = None
    attacks = heuristics.best_attacks(board, turn)
    for source, target, prob in attacks:
        p_succ = prob
        p_fail = 1. - prob
        with Attack(source, target, True):
            v_succ, _ = expectimax_n(board, depth - 1, next_turn, n, heuristics)
        with Attack(source, target, False):
            v_fail, _ = expectimax_n(board, depth - 1, next_turn, n, heuristics)
        val = (p_succ * v_succ) + (p_fail * v_fail)
        if val[turn] > best_val[turn]:
            best_val = val
            best_act = source.get_name(), target.get_name()
    val, _ = expectimax_n(board, depth - 1, next_turn, n, heuristics)
    if val[turn] > best_val[turn]:
        best_val = val
        best_act = None
    return best_val, best_act
