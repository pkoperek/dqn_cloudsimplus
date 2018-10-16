import torch
from torch import nn
from torchviz import make_dot
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(6, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm1d(32)
        self.head = nn.Linear(32 * 222, 3)

    def forward(self, x):
        x = F.selu(self.bn1(self.conv1(x)))
        x = F.selu(self.bn2(self.conv2(x)))
        x = F.selu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


x = torch.randn(1, 6, 1800)

model = DQN().to('cpu')

dot = make_dot(model(x), params=dict(model.named_parameters()))
dot.format = 'png'
dot.render('dqn_model.gv', view=True)
