import torch
from torch import nn
from torch.nn import functional as F

class ConcatModel(nn.Module):
    def __init__(self, model, out_channels, num_classes):
        super().__init__()
        self.cnn = model

        self.fc1 = nn.Linear(out_channels+2, int((out_channels+2)/2))
        self.fc2 = nn.Linear(int((out_channels+2)/2), num_classes)

    def forward(self, image, meta):
        x1 = self.cnn(image)
        x2 = meta

        x = torch.cat((x1,x2), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x