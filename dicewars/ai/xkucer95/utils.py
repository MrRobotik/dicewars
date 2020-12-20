from dicewars.client.game.board import Board
from dicewars.client.game.board import Area
import numpy as np

ATTACK_SUCC_PROBS = {
    2: {
        1: 0.83796296,
        2: 0.44367284,
        3: 0.15200617,
        4: 0.03587963,
        5: 0.00610497,
        6: 0.00076625,
        7: 0.00007095,
        8: 0.00000473,
    },
    3: {
        1: 0.97299383,
        2: 0.77854938,
        3: 0.45357510,
        4: 0.19170096,
        5: 0.06071269,
        6: 0.01487860,
        7: 0.00288998,
        8: 0.00045192,
    },
    4: {
        1: 0.99729938,
        2: 0.93923611,
        3: 0.74283050,
        4: 0.45952825,
        5: 0.22044235,
        6: 0.08342284,
        7: 0.02544975,
        8: 0.00637948,
    },
    5: {
        1: 0.99984997,
        2: 0.98794010,
        3: 0.90934714,
        4: 0.71807842,
        5: 0.46365360,
        6: 0.24244910,
        7: 0.10362599,
        8: 0.03674187,
    },
    6: {
        1: 0.99999643,
        2: 0.99821685,
        3: 0.97529981,
        4: 0.88395347,
        5: 0.69961639,
        6: 0.46673060,
        7: 0.25998382,
        8: 0.12150697,
    },
    7: {
        1: 1.00000000,
        2: 0.99980134,
        3: 0.99466336,
        4: 0.96153588,
        5: 0.86237652,
        6: 0.68516499,
        7: 0.46913917,
        8: 0.27437553,
    },
    8: {
        1: 1.00000000,
        2: 0.99998345,
        3: 0.99906917,
        4: 0.98953404,
        5: 0.94773146,
        6: 0.84387382,
        7: 0.67345564,
        8: 0.47109073,
    }
}


def possible_attacks(board: Board, player_name: int):
    for source in board.get_player_border(player_name):
        if not source.can_attack():
            continue
        for adj in source.get_adjacent_areas():
            target = board.get_area(adj)
            if target.get_owner_name() != player_name:
                succ_prob = ATTACK_SUCC_PROBS[source.get_dice()][target.get_dice()]
                yield source, target, succ_prob


def state_descriptor(board: Board, player_name: int, players: list):
    areas = board.get_player_areas(player_name)
    regions = board.get_players_regions(player_name)
    border = board.get_player_border(player_name)
    max_region_size = max(len(r) for r in regions)
    rel_border_size_1 = len(border) / sum(len(board.get_player_border(name)) for name in players)
    rel_border_size_2 = sum(a.get_dice() for a in border) / sum(a.get_dice() for a in areas)

    best_border = []
    for r in regions:
        if len(r) == max_region_size:
            for area in map(lambda a: board.get_area(a), r):
                if board.is_at_border(area):
                    best_border.append(area)
    border_strength = 0.
    for target in best_border:
        survival_prob = 1.
        for adj in target.get_adjacent_areas():
            source = board.get_area(adj)
            if source.get_owner_name() == player_name or not source.can_attack():
                continue
            survival_prob *= 1. - ATTACK_SUCC_PROBS[source.get_dice()][target.get_dice()]
        border_strength += survival_prob
    border_strength /= len(best_border)

    neigh_dice_dist = []
    for neigh in map(lambda a: a.get_adjacent_areas(), areas):
        player_power = 0
        total_power = 0
        for area in map(lambda a: board.get_area(a), neigh):
            dice = area.get_dice()
            if area.get_owner_name() == player_name:
                player_power += dice
            total_power += dice
        neigh_dice_dist.append(player_power / total_power)

    feature_vector = [
        max_region_size,
        rel_border_size_1,
        rel_border_size_2,
        border_strength,
        np.mean(neigh_dice_dist),
    ]
    eps = np.finfo(np.float32).eps.item()
    return np.asarray(feature_vector, dtype=np.float32) + eps


def standardize_data(x: np.ndarray, axis=0):
    eps = np.finfo(x.dtype).eps.item()
    return (x - np.mean(x, axis=axis)) / (np.std(x, axis=axis) + eps)
