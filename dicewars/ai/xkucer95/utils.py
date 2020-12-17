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


def make_attack_descriptor(board: Board, source: Area, target: Area):
    player_name = source.get_owner_name()
    succ_prob = ATTACK_SUCC_PROBS[source.get_dice()][target.get_dice()]
    source_power = 0
    target_power = 0
    with Attack(source, target, True):
        for adj in source.get_adjacent_areas():
            area = board.get_area(adj)
            if area.get_owner_name() == player_name:
                source_power += area.get_dice()
            else:
                source_power -= area.get_dice()
        for adj in target.get_adjacent_areas():
            area = board.get_area(adj)
            if area.get_owner_name() == player_name:
                target_power += area.get_dice()
            else:
                target_power -= area.get_dice()
        regions = board.get_players_regions(player_name)
    best_region_size = max(map(lambda x: len(x), regions))
    region_size = None
    for region in regions:
        if source.get_name() in region:
            region_size = len(region)
            break
    best_region_bit = (region_size == best_region_size)
    feature_vector = [
        succ_prob,
        best_region_bit,
        source_power,
        target_power,
        region_size,
    ]
    return np.asarray(feature_vector)


def standardize_data(x: np.ndarray, axis: int):
    eps = np.finfo(x.dtype).eps.item()
    return (x - np.mean(x, axis=axis)) / (np.std(x, axis=axis) + eps)


class Attack:
    def __init__(self, source: Area, target: Area, succ: bool):
        self.source = source
        self.target = target
        self.source_dice = source.get_dice()
        self.source_owner = source.get_owner_name()
        self.target_dice = target.get_dice()
        self.target_owner = target.get_owner_name()
        self.succ = succ

    def __enter__(self):
        self.source.set_dice(1)
        if self.succ:
            self.target.set_dice(self.source_dice - 1)
            self.target.set_owner(self.source_owner)

    def __exit__(self, *args):
        self.source.set_dice(self.source_dice)
        if self.succ:
            self.target.set_dice(self.target_dice)
            self.target.set_owner(self.target_owner)
