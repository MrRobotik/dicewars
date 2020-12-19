import torch.nn
import torch.nn.functional as f
import torch.distributions
import numpy as np


class PolicyModel(torch.nn.Module):
    def __init__(self, on_policy: bool):
        super().__init__()
        self.affine1 = torch.nn.Linear(12, 6, True)
        self.affine2 = torch.nn.Linear( 6, 2, True)
        self.dropout = torch.nn.Dropout(0.2)
        self.probs_buff = []
        self.train(on_policy)
        self.on_policy = on_policy

    def forward(self, x):
        a = self.affine1(x)
        a = self.dropout(a)
        a = f.relu(a)
        a = self.affine2(a)
        y = f.softmax(a)
        return y

    def select_action(self, data_in: np.ndarray):
        attacks, probs = [], []
        for x in data_in:
            y = self(torch.from_numpy(x.astype(np.float32)))
            if self.on_policy:
                m = torch.distributions.Categorical(y)
                action = int(m.sample())
            else:
                action = int(np.argmax(y))
            if action == 1:
                attacks.append(action)
                probs.append(y[action])
        if len(attacks) == 0:
            return None
        best_attack = int(np.argmax(probs))
        prob = probs[best_attack]
        self.probs_buff.append(prob)
        return best_attack

    def calc_grads(self, reward: float):
        if reward >= 0.:
            probs = (p for p in self.probs_buff)
        else:
            probs = (1. - p for p in self.probs_buff)
        reward = abs(reward) / len(self.probs_buff)
        eps = np.finfo(np.float32).eps.item()
        probs = (p + eps if p == 0. else p for p in probs)
        loss = sum(-torch.log(p) * reward for p in probs)
        loss.backward()
        self.probs_buff.clear()

        loss_output = open('dicewars/ai/xkucer95/models/loss.csv', 'a')
        loss_output.write('{} {}\n'.format(float(loss), reward))
        loss_output.close()
