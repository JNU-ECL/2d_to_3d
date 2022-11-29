import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from smplx.body_models import SMPL

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        # self.conv2d = nn.Conv2d(3, 3, 3, stride=1)
        self.resnet = models.resnet50() 
        self.FC = nn.Linear(1000, 82)

    def forward(self, x):
        # x = F.relu(self.conv2d(x))
        x = F.relu(self.resnet(x))
        x = torch.sigmoid(self.FC(x))
        return x

