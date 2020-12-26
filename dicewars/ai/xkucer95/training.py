import torch.optim
import torch.nn.functional
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy

from dicewars.ai.xkucer95.happ_model import HoldAreaProbPredictor
from dicewars.ai.xkucer95.utils import batch_provider, evaluate


def main():
    epochs = 0
    lr = 0.
    x_trn = None
    x_val = None
    t_trn = None
    t_val = None
    try:
        data_dir = sys.argv[1]
        model_name = sys.argv[2]
        epochs = int(sys.argv[3])
        lr = float(sys.argv[4])
        x_trn = np.loadtxt('{}/{}_x_trn.csv'.format(data_dir, model_name), delimiter=' ').astype(np.float32)
        x_val = np.loadtxt('{}/{}_x_val.csv'.format(data_dir, model_name), delimiter=' ').astype(np.float32)
        t_trn = np.loadtxt('{}/{}_t_trn.csv'.format(data_dir, model_name), delimiter=' ').astype(np.float32)
        t_val = np.loadtxt('{}/{}_t_val.csv'.format(data_dir, model_name), delimiter=' ').astype(np.float32)
    except:
        exit(1)

    model = HoldAreaProbPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batch_size = 32

    best_accuracy = 0.
    best_model = copy.deepcopy(model)
    losses = []
    accuracies = []

    for epoch in range(epochs):
        loss_avg = 0.
        for x, t in batch_provider(x_trn, t_trn, batch_size):
            y = model(x)
            loss = torch.nn.functional.binary_cross_entropy(y, t)
            loss.backward()
            optimizer.step()
            loss_avg += float(loss) * (1/len(t))
        with torch.no_grad():
            accuracy = evaluate(model, x_val, t_val)
        losses.append(loss_avg)
        accuracies.append(accuracy)
        if accuracy > best_accuracy:
            print(accuracy)
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)

    torch.save(best_model.state_dict(), best_model.model_path)
    print('best acc.: {}'.format(best_accuracy))
    plt.subplot(2, 1, 1)
    plt.plot(losses, label='loss')
    plt.legend(loc='upper right')
    plt.subplot(2, 1, 2)
    plt.plot(accuracies, label='accuracy')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
