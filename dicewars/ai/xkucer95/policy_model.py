import torch.nn
import torch.distributions
import numpy as np
from os import path


class PolicyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.affine_atk = torch.nn.Linear(3, 1, True)
        self.affine_cst = torch.nn.Linear(6, 3, True)
        self.affine_nst = torch.nn.Linear(6, 3, True)
        self.affine_cmp = torch.nn.Linear(6, 1, True)
        self.affine_fin = torch.nn.Linear(2, 1, True)
        self.model_path = 'dicewars/ai/xkucer95/models/policy_model.pt'
        if path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path))

    def forward(self, x):
        if x.ndim == 1:
            x_atk = x[0:3]
            x_cst = x[3:9]
            x_nst = x[9:]
        else:
            x_atk = x[:, 0:3]
            x_cst = x[:, 3:9]
            x_nst = x[:, 9:]

        a_atk = torch.sigmoid(self.affine_atk(x_atk))
        a_cst = torch.sigmoid(self.affine_cst(x_cst))
        a_nst = torch.sigmoid(self.affine_cst(x_nst))
        a_cmp = torch.sigmoid(self.affine_cmp(torch.cat((a_cst, a_nst), dim=x.ndim-1)))
        a_fin = torch.sigmoid(self.affine_fin(torch.cat((a_atk, a_cmp), dim=x.ndim-1)))
        return a_fin

    def select_action(self, data_in: np.ndarray, sample=False):
        probs = []
        # print('-----------------------------------------')
        for x in data_in:
            with torch.no_grad():
                y = self(torch.from_numpy(x))
                if sample:
                    m = torch.distributions.Bernoulli(probs=y)
                    if int(m.sample()) == 1:
                        probs.append(y)
                else:
                    probs.append(y)
                # print('prob: ', y)
        if len(probs) == 0:
            return None
        best_attack = int(np.argmax(probs))
        return best_attack

    def backward(self):
        loss = sum(-torch.log(p) * r for p, r in self.train_buff)
        loss /= len(self.train_buff)
        loss.backward()
        self.train_buff.clear()
