from .utils import *


class Action:
    def __init__(self, board: Board):
        self.board = board
        self.stack = []
        self.succ_prob = 1.

    def attack(self, source: Area, target: Area, succ_prob: float, succ_flag: bool):
        s = (source, source.get_dice(), source.get_owner_name())
        t = (target, target.get_dice(), target.get_owner_name())
        self.stack.append((s, t, succ_flag))
        s[0].set_dice(1)
        if succ_flag:
            self.succ_prob *= succ_prob
            t[0].set_dice(s[1] - 1)
            t[0].set_owner(s[2])
        else:
            self.succ_prob *= 1. - succ_prob

    def rollback(self):
        while self.stack:
            s, t, succ_flag = self.stack.pop()
            s[0].set_dice(s[1])
            if succ_flag:
                t[0].set_dice(t[1])
                t[0].set_owner(t[2])
        self.succ_prob = 1.
