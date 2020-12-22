import torch.nn
import torch.distributions
from os import path


class HoldAreaProbPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = torch.nn.Linear(6, 9, True)
        self.affine2 = torch.nn.Linear(9, 1, True)
        self.model_path = 'dicewars/ai/xkucer95/models/happ_model.pt'
        if path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path))

    def forward(self, x):
        a = torch.sigmoid(self.affine1(x))
        y = torch.sigmoid(self.affine2(a))
        return y
