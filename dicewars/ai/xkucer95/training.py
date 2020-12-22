import torch.optim
import torch.nn.functional
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy

from dicewars.ai.xkucer95.happ_model import HoldAreaProbPredictor
from dicewars.ai.xkucer95.utils import batch_provider, evaluate


def main():
    x_trn = None
    x_eval = None
    t_trn = None
    t_eval = None
    try:
        data_dir = sys.argv[1]
        model_name = sys.argv[2]
        x_trn = np.loadtxt('{}/{}_x_trn.csv'.format(data_dir, model_name), delimiter=',').astype(np.float32)
        x_eval = np.loadtxt('{}/{}_x_eval.csv'.format(data_dir, model_name), delimiter=',').astype(np.float32)
        t_trn = np.loadtxt('{}/{}_t_trn.csv'.format(data_dir, model_name), delimiter=',').astype(np.float32)
        t_eval = np.loadtxt('{}/{}_t_eval.csv'.format(data_dir, model_name), delimiter=',').astype(np.float32)
    except:
        exit(1)

    model = HoldAreaProbPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch_size = 32

    best_accuracy = 0.
    best_model = copy.deepcopy(model)
    losses = []
    accuracies = []

    for epoch in range(100):
        loss_avg = 0.
        for x, t in batch_provider(x_trn, t_trn, batch_size):
            y = model(x)
            loss = torch.nn.functional.binary_cross_entropy(y, t)
            loss.backward()
            optimizer.step()
            loss_avg += float(loss) * (1/len(t))
        with torch.no_grad():
            accuracy = evaluate(model, x_eval, t_eval)
        losses.append(loss_avg)
        accuracies.append(accuracy)
        print(loss_avg)
        print('acc:', accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)

    torch.save(best_model.state_dict(), best_model.model_path)
    plt.plot(losses, label='loss')
    plt.plot(accuracies, label='accuracy')
    plt.show()


if __name__ == '__main__':
    main()
