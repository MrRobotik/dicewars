import torch.nn
import torch.nn.functional as f
import torch.distributions
import numpy as np


class PolicyModel(torch.nn.Module):
    def __init__(self, on_policy: bool):
        super().__init__()
        self.affine1 = torch.nn.Linear(6, 3, True)
        self.affine2 = torch.nn.Linear(3, 1, True)
        self.train(on_policy)
        self.on_policy = on_policy
        if on_policy:
            self.train_temp = list()
            self.train_buff = list()

    def forward(self, x):
        a = self.affine1(x)
        a = f.sigmoid(a)
        a = self.affine2(a)
        y = f.sigmoid(a)
        return y

    def select_action(self, data_in: np.ndarray):
        probs = []
        for x in data_in:
            if self.on_policy:
                y = self(torch.from_numpy(x))
                m = torch.distributions.Bernoulli(probs=y)
                action = int(m.sample())
            else:
                with torch.no_grad():
                    y = self(torch.from_numpy(x))
                    action = 1 if y > 0.5 else 0
            if action == 1:
                probs.append(y)
        if len(probs) == 0:
            return None
        best_attack = int(np.argmax(probs))
        if self.on_policy:
            self.train_temp.append(probs[best_attack])
        return best_attack

    def give_reward(self, reward):
        discount = 0.9
        n = 0
        while self.train_temp:
            p = self.train_temp.pop()
            p = 1. - p if reward < 0. else p
            r = discount**n * abs(reward)
            if p == 0.:
                p += np.finfo(np.float32).eps.item()
            self.train_buff.append((p, r))
            n += 1

    def backward(self):
        loss = sum(-torch.log(p) * r for p, r in self.train_buff)
        loss /= len(self.train_buff)
        loss.backward()
        self.train_buff.clear()

        loss_output = open('dicewars/ai/xkucer95/models/loss.csv', 'a')
        loss_output.write('{}\n'.format(float(loss)))
        loss_output.close()
