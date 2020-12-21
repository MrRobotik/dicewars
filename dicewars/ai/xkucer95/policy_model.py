import torch.nn
import torch.distributions
import numpy as np
from os import path


class PolicyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logistic = torch.nn.Linear(15, 1, True)
        # self.affine_atk_1 = torch.nn.Linear(3, 1, True)
        # self.affine_cst = torch.nn.Linear(15, 10, True)
        # self.affine_nst = torch.nn.Linear(10,  1, True)
        self.model_path = 'dicewars/ai/xkucer95/models/policy_model.pt'
        if path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path))

    def forward(self, x):
        # x1 = torch.from_numpy(x[0:3])
        # x2 = torch.from_numpy(x[3:9])
        # x3 = torch.from_numpy(x[9:])
        y = torch.sigmoid(self.logistic(torch.from_numpy(x)))
        return y

    def select_action(self, data_in: np.ndarray, sample=False):
        probs = []
        # print('-----------------------------------------')
        for x in data_in:
            with torch.no_grad():
                y = self(x)
                if sample:
                    m = torch.distributions.Bernoulli(probs=y)
                    action = int(m.sample())
                else:
                    action = 1 if y > 0.5 else 0
                # print('prob: ', y)
            if action == 1:
                probs.append(y)
        if len(probs) == 0:
            return None
        best_attack = int(np.argmax(probs))
        return best_attack

    def backward(self):
        loss = sum(-torch.log(p) * r for p, r in self.train_buff)
        loss /= len(self.train_buff)
        loss.backward()
        self.train_buff.clear()
