import torch
from torch import nn
from torch.nn import functional as F

class ConcatModel2(nn.Module):
    def __init__(self, model, out_channels, num_classes):
        super().__init__()
        self.cnn = model

        self.meta_fc1 = nn.Linear(2, 2)
        self.meta_fc2 = nn.Linear(2, 2)

        # self.concat_fc1 = nn.Linear(out_channels+2, int((out_channels+2)/2))
        self.concat_fc2 = nn.Linear(out_channels+2, num_classes)

    def forward(self, image, meta):
        x1 = self.cnn(image)
        x2 = meta

        x2 = F.relu(self.meta_fc1(x2))
        x2 = self.meta_fc2(x2)

        x = torch.cat((x1,x2), dim=1)
        # x = F.relu(self.concat_fc1(x))
        x = self.concat_fc2(x)

        return x