import torch.nn
import torch.distributions
import numpy as np
from os import path


class PolicyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = torch.nn.Linear(15, 3, True)
        self.affine2 = torch.nn.Linear( 3, 1, True)
        self.model_path = 'dicewars/ai/xkucer95/models/policy_model.pt'
        if path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path))

    def forward(self, x):
        a = self.affine1(x)
        a = torch.sigmoid(a)
        a = self.affine2(a)
        y = torch.sigmoid(a)
        return y

    def select_action(self, data_in: np.ndarray, sample=False):
        probs = []
        # print('-----------------------------------------')
        for x in data_in:
            with torch.no_grad():
                y = self(torch.from_numpy(x))
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
