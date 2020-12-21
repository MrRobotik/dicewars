import torch.optim
import numpy as np
import sys

from dicewars.ai.xkucer95.policy_model import PolicyModel


def batch_provider(trn, batch_size):
    x = np.random.permutation(trn)
    for i in range(0, x.shape[0] - batch_size, batch_size):
        yield x[i:i+batch_size, :-1], x[i:i+batch_size, -1]


def main():
    trn = None
    try:
        trn = np.loadtxt(sys.argv[1]).astype(np.float32)
    except:
        exit(1)

    policy_model = PolicyModel()
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-3)
    batch_size = 128

    for epoch in range(20):
        total_loss = 0.
        for x, r in batch_provider(trn, batch_size):
            y = policy_model(torch.from_numpy(x))
            pos = np.where(r > 0.)
            neg = np.where(r < 0.)
            optimizer.zero_grad()
            loss1 = torch.sum(-torch.log(y[pos]) * torch.from_numpy(np.abs(r[pos])))
            loss2 = torch.sum(-torch.log(1. - y[neg]) * torch.from_numpy(np.abs(r[neg])))
            loss = loss1 + loss2
            loss /= batch_size
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
        print(total_loss / batch_size)

    torch.save(policy_model.state_dict(), policy_model.model_path)


if __name__ == '__main__':
    main()
