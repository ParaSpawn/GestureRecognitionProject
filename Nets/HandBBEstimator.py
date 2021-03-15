import torchvision
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import random
from PIL import Image
from PIL import ImageDraw


class Res2d(nn.Module):

    def __init__(self, channels):
        super(Res2d, self).__init__()
        self.c1 = nn.Conv2d(channels, channels, 1)
        self.c2 = nn.Conv2d(channels, channels, 1)

    def forward(self, t):
        i = t
        t = F.relu(self.c1(t))
        t = F.relu(self.c2(t) + i)
        return t


class HandBBEstimator(nn.Module):

    def __init__(self):
        super(HandBBEstimator, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, 5, 2)
        self.res1 = Res2d(9)
        self.conv2 = nn.Conv2d(9, 18, 3, 2)
        self.res2 = Res2d(18)
        self.conv3 = nn.Conv2d(18, 36, 3, 2)
        self.res3 = Res2d(36)
        self.conv4 = nn.Conv2d(36, 74, 3, 2)
        self.conv5 = nn.Conv2d(74, 5, 1)

    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = self.res1(t)
        t = F.relu(self.conv2(t))
        t = self.res2(t)
        t = F.relu(self.conv3(t))
        t = self.res3(t)
        t = F.relu(self.conv4(t))
        t = F.relu(self.conv5(t))
        return t
