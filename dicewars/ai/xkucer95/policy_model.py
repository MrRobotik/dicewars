import torch.nn
import torch.nn.functional as f
import torch.distributions
import numpy as np


class PolicyModel(torch.nn.Module):
    def __init__(self, on_policy: bool):
        super().__init__()
        self.affine1 = torch.nn.Linear(12, 6, True)
        self.affine2 = torch.nn.Linear( 6, 2, True)
        self.train(on_policy)
        self.on_policy = on_policy
        if on_policy:
            self.train_buff_1 = []
            self.train_buff_2 = []

    def forward(self, x):
        a = self.affine1(x)
        a = f.relu(a)
        a = self.affine2(a)
        y = f.softmax(a)
        return y

    def select_action(self, data_in: np.ndarray):
        attacks, probs = [], []
        for x in data_in:
            if self.on_policy:
                y = self(torch.from_numpy(x.astype(np.float32)))
                m = torch.distributions.Categorical(y)
                action = int(m.sample())
            else:
                with torch.no_grad():
                    y = self(torch.from_numpy(x.astype(np.float32)))
                    action = int(np.argmax(y))
            if action == 1:
                attacks.append(action)
                probs.append(y[action])
        if len(attacks) == 0:
            return None
        best_attack = int(np.argmax(probs))
        prob = probs[best_attack]
        if self.on_policy:
            self.train_buff_2.append(prob)
        return best_attack

    def reward_for_turn(self, reward):
        discount = 0.9
        n = 0
        while self.train_buff_2:
            p = self.train_buff_2.pop()
            p = 1. - p if reward < 0. else p
            r = discount**n * abs(reward)
            if p == 0.:
                p += np.finfo(np.float32).eps.item()
            self.train_buff_1.append((p, r))
            n += 1

    def backward(self):
        loss = sum(-torch.log(p) * r for p, r in self.train_buff_1)
        loss /= len(self.train_buff_1)
        loss.backward()
        self.train_buff_1.clear()

        loss_output = open('dicewars/ai/xkucer95/models/loss.csv', 'a')
        loss_output.write('{}\n'.format(float(loss)))
        loss_output.close()
