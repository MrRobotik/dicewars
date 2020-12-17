import torch.nn
import torch.nn.functional
import torch.distributions


class PolicyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = torch.nn.Linear(7, 7, True)
        self.affine2 = torch.nn.Linear(7, 1, True)
        # self.dropout = torch.nn.Dropout(0.2, False)
        self.buffer = []
        self.log_prob = None

    def forward(self, x):
        y = self.affine1(x)
        y = self.affine1(torch.nn.functional.relu(y))
        self.buffer.append(y)

    def sample_action(self, backward=False):
        y = torch.stack(tuple(self.buffer))
        probs = torch.nn.functional.softmax(y)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log_prob = m.log_prob(action)
        self.buffer.clear()
        return action

    def backward(self, reward):
        loss = -self.log_prob * reward
        loss.backward()
