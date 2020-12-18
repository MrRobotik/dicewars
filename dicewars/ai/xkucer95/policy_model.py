import torch.nn
import torch.nn.functional
import torch.distributions


class PolicyModel(torch.nn.Module):
    def __init__(self, train_online):
        super().__init__()
        self.affine1 = torch.nn.Linear(5, 5, True)
        self.affine2 = torch.nn.Linear(5, 1, True)
        self.log_probs_buff = []
        self.train(train_online)
        self.train_online = train_online

    def forward(self, x):
        y = self.affine1(x)
        y = torch.nn.functional.sigmoid(y)
        y = self.affine2(y)
        return y

    def forward_all(self, x):
        for i in range(x.shape[0]):
            yield self(torch.from_numpy(x[i, :]))

    def select_action(self, x):
        y = torch.cat(tuple(self.forward_all(x)))
        probs = torch.nn.functional.softmax(y)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        if self.train_online:
            self.log_probs_buff.append(m.log_prob(action))
        return action

    def calc_grads(self, reward):
        loss = sum(-log_prob * reward for log_prob in self.log_probs_buff)
        loss.backward()
        self.log_probs_buff.clear()
