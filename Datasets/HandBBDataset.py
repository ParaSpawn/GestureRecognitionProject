import torchvision
import torchvision.transforms as transforms
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import csv
import os
import operator
from PIL import Image
from PIL import ImageDraw
import random
import matplotlib.patches as patches

class HandBoundingBoxDataset(T.utils.data.Dataset):
    def __init__(self, csv_path, root_path, scales, device, transform=None):
        self.root_path = root_path
        self.csv_data = pd.read_csv(csv_path)
        self.transform = transform
        self.device = device
        self.scales = scales

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, i):
        img_name = self.root_path + self.csv_data.iloc[i, 0]
        image = Image.open(img_name)
        box = np.array(self.csv_data.iloc[i, 1:])
        coord = [box[0] + box[2] / 2, box[1] + box[3] / 2]
        size = images.size()
        images = []
        for scale in self.scales:
            t = transforms.Compose([transforms.Resize((scale * size[0], scale * size[1])), transforms.ToTensor()])
            images.append(t(image).to(self.device))
        return images, coord

class HandBoundingBoxDataset2(T.utils.data.Dataset):
    def __init__(self, csv_path, root_path, device, scaling_factor=0.65, stride=32, transform=None):
        self.root_path = root_path
        self.csv_data = pd.read_csv(csv_path)
        self.transform = transform
        self.device = device
        self.img_size = [480, 640]
        self.scaling_factor = scaling_factor
        self.stride = stride

    def __len__(self):
        return len(self.csv_data)

    def get_grid_dim(self):
        return [self.img_size[1] * self.scaling_factor // self.stride, self.img_size[0] * self.scaling_factor // self.stride]

    def __getitem__(self, i):
        img_name = self.root_path + self.csv_data.iloc[i, 0]
        img_name2 = self.root_path + self.csv_data.iloc[i + 1, 0]
        img_name3 = self.root_path + self.csv_data.iloc[i + 2, 0]
        img_name4 = self.root_path + self.csv_data.iloc[i + 3, 0]
        image = Image.open(img_name)
        image2 = Image.open(img_name2)
        image3 = Image.open(img_name3)
        image4 = Image.open(img_name4)
        random.seed(a=i)
        box = np.array(self.csv_data.iloc[i, 1:])
        width, height = image.size
        height *= self.scaling_factor
        width *= self.scaling_factor
        grid_x, grid_y = width // self.stride, height // self.stride
        for i in range(len(box)):
            box[i] *= self.scaling_factor
        box[0] += box[2] / 2
        box[1] += box[3] / 2
        gx, gy = int(box[0] // self.stride), int(box[1] // self.stride)
        ret = T.FloatTensor([(box[0] / self.stride - gx) * 100, (box[1] / self.stride - gy) * 100, box[2] / 2, box[3] / 2, 100, 100]).to(self.device)
        t = transforms.Compose([transforms.Resize((int(height), int(width)))])
        image = t(image)
        image2 = t(image2)
        image3 = t(image3)
        image4 = t(image4)
        img = Image.new('RGB', (416 * 2, 312 * 2))
        img.paste(image, (0, 0))
        img.paste(image2, (416, 0))
        img.paste(image3, (0, 312))
        img.paste(image4, (416, 312))
        t = transforms.Compose([transforms.ToTensor()])
        img = t(img).transpose(1, 2).to(self.device)
        return img, gx, gy, ret