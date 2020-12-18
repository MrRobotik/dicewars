import torch.nn
import torch.nn.functional
import torch.distributions
import numpy as np


class PolicyModel(torch.nn.Module):
    def __init__(self, train_online):
        super().__init__()
        self.affine1 = torch.nn.Linear(6, 32, True)
        self.affine2 = torch.nn.Linear(32, 1, True)
        self.dropout = torch.nn.Dropout(0.2)
        self.probs_buff = []
        self.train(train_online)
        self.train_online = train_online

    def forward(self, x):
        y = self.affine1(x)
        y = self.dropout(y)
        y = torch.nn.functional.relu(y)
        y = self.affine2(y)
        return y

    def forward_all(self, data_in):
        for x in data_in:
            yield self(torch.from_numpy(x.astype(np.float32)))

    def select_action(self, data_in):
        y = torch.cat(tuple(self.forward_all(data_in)))
        probs = torch.nn.functional.softmax(y)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        if self.train_online:
            self.probs_buff.append(probs[action])
        return action

    def calc_grads(self, reward):
        if reward >= 0.:
            probs = (p for p in self.probs_buff)
        else:
            probs = (1. - p for p in self.probs_buff)
        eps = np.finfo(np.float32).eps.item()
        probs = (p + eps if p == 0. else p for p in probs)
        loss = sum(-torch.log(p) * abs(reward) for p in probs)
        loss.backward()
        self.probs_buff.clear()

        loss_output = open('/home/adam/Documents/dicewars/dicewars/ai/xkucer95/models/loss.csv', 'a')
        loss_output.write(str(float(loss)) + '\n')
        loss_output.close()
