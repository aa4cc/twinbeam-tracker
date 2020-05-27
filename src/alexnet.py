import torch
import torch.nn.functional as F
from torch import nn

from .config import config


class SiameseAlexNet(nn.Module):
    def __init__(self, gpu_id):
        super(SiameseAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 32, 3, 1, groups=2)
        )
        self.corr_bias = nn.Parameter(torch.zeros(1))
        self.exemplar = None
        self.gpu_id = gpu_id

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        exemplar, instance = x
        if exemplar is not None and instance is not None:
            batch_size = exemplar.shape[0]
            exemplar = self.features(exemplar)
            instance = self.features(instance)
            N, C, H, W = instance.shape
            instance = instance.view(1, -1, H, W)
            score = F.conv2d(instance, exemplar, groups=N) * config.response_scale \
                    + self.corr_bias
            return score.transpose(0, 1)
        elif exemplar is not None and instance is None:
            # inference used
            self.exemplar = self.features(exemplar)
        else:
            # inference used we don't need to scale the reponse or add bias
            instance = self.features(instance)
            N, _, H, W = instance.shape
            if (self.exemplar.shape[0] != N) and (self.exemplar.shape[0] == 1):
                self.exemplar = torch.cat([self.exemplar for _ in range(N)], dim=0)
            instance = instance.view(1, -1, H, W)
            score = F.conv2d(instance, self.exemplar, groups=N)
            return score.transpose(0, 1)
