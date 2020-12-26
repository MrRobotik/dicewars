import torch.nn
from os import path


class WinProbPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logistic = torch.nn.Linear(6, 1, True)
        self.model_path = 'dicewars/ai/xkucer95/models/wpp_model.pt'
        if path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path))

    def forward(self, x):
        y = torch.sigmoid(self.logistic(x))
        return y
