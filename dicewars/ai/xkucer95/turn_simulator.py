from .utils import *


class TurnSimulator:
    def __init__(self, board: Board):
        self.board = board
        self.stack = []
        self.succ_prob = 1.

    def __del__(self):
        while self.stack:
            self.undo_attack()

    def do_attack(self, source: Area, target: Area, succ_prob: float, succ_flag: bool):
        s_dice, s_owner = source.get_dice(), source.get_owner_name()
        t_dice, t_owner = target.get_dice(), target.get_owner_name()
        self.stack.append((source, s_dice, s_owner, target, t_dice, t_owner, self.succ_prob, succ_flag))
        source.set_dice(1)
        if succ_flag:
            self.succ_prob *= succ_prob
            target.set_dice(s_dice - 1)
            target.set_owner(s_owner)
        else:
            self.succ_prob *= 1. - succ_prob

    def undo_attack(self):
        source, s_dice, s_owner, target, t_dice, t_owner, self.succ_prob, succ_flag = self.stack.pop()
        source.set_dice(s_dice)
        if succ_flag:
            target.set_dice(t_dice)
            target.set_owner(t_owner)
