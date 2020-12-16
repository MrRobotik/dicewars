from .utils import *
import numpy as np


def evaluation(board: Board, players_order: list):
    val = []
    for player in players_order:
        val.append(max(len(region) for region in board.get_players_regions(player)))
    return np.asarray(val)


def expectimax_n(board: Board, depth: int, turn: int, players_order: list):
    if board.nb_players_alive() == 1 or depth == 0:
        return evaluation(board, players_order), None
    n = len(players_order)
    best_val = np.full(len(players_order), -np.infty)
    best_act = None
    attacks = possible_attacks(board, players_order[turn])
    attacks = sorted(attacks, key=lambda x: x[2], reverse=True)[:2]
    for source, target, prob in attacks:
        p_succ = prob
        p_fail = 1. - prob
        with Attack(source, target, True):
            v_succ, _ = expectimax_n(board, depth-1, (turn+1) % n, players_order)
        with Attack(source, target, False):
            v_fail, _ = expectimax_n(board, depth-1, (turn+1) % n, players_order)
        val = (p_succ * v_succ) + (p_fail * v_fail)
        if val[turn] > best_val[turn]:
            best_val = val
            best_act = source.get_name(), target.get_name()
    val, _ = expectimax_n(board, depth-1, (turn+1) % n, players_order)
    if val[turn] > best_val[turn]:
        best_val = val
        best_act = None
    return best_val, best_act
