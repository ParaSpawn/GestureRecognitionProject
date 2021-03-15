import json
from torch import autograd
import Nets.GenericNet as GN
import Datasets.HandPoseDataset as HPD
from PIL import ImageDraw
from PIL import Image
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch as T
import torchvision
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms

sys.path.insert(1, 'D:/projects/GestureRecognition/')

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
print(T.cuda.get_device_name(device))

scales = [1.5, 1.0, 0.75, 0.5]
data_set = HPD.HandBoundingBoxDataset(
    HPD.PATHS['hpd_bbox_data'], HPD.PATHS['hpd_images'], scales, device)


def create_loaders(dataset, batch_size, training_split, shuffle, seed):
    indices = list(range(len(dataset)))
    split = int(np.floor(training_split * len(dataset)))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[:split], indices[split:]

    train_sampler = T.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = T.utils.data.SubsetRandomSampler(val_indices)
    train_loader = T.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
    validation_loader = T.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=valid_sampler)
    return train_loader, validation_loader


batch_size = 5
check_point = 200
learning_rate = 0.001
train_loader, test_loader = create_loaders(
    data_set, batch_size, 0.3, True, 330)

model_name = 'HANDDETECTOR'
net = GN.GenericNet(GN.PATHS[model_name], device).to(device)
net.load_state_dict(T.load(GN.WEIGHT_PATHS[model_name]))
net.train()

criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

floatTensor = T.cuda.FloatTensor

try:
    with open(HPD.JSON_PATHS[model_name], 'x') as f:
        with open(HPD.JSON_PATHS[model_name], 'w') as fi:
            data = {'all_loss': []}
            json.dump(data, fi)
except:
    print('file already exists')

all_loss = []

# scale, batch, color channel, height, width
# scale, batch, (conf, offset_x, offset_y, size), height, width

with T.enable_grad():
    for epoch in range(1):
        for i, d in enumerate(train_loader, 0):
            images, coords = d
            for j, scale in enumerate(scales):
                op = net(images[j])[-1]
                target = op.clone().detach()
                # target = T.zeros(*op.shape).to(device)
                for k in range(batch_size):
                    x, y, w = coords[k]
                    cell_y = int(y * target.size(2))
                    cell_x = int(x * target.size(3))
                    try:
                        target[k, 0, :, :] = 0
                        target[k, 0, cell_y, cell_x] = 10
                        target[k, 1, cell_y, cell_x] = (x * target.size(3) - cell_x) * 10
                        target[k, 2, cell_y, cell_x] = (y * target.size(3) - cell_y) * 10
                        target[k, 3, cell_y, cell_x] = w * 10
                    except:
                        continue
                optimizer.zero_grad()
                loss = criterion(op, target)
                loss.backward()
                optimizer.step()
            print(i, loss.item())
            all_loss.append(loss.item())
            if (i + 1) % check_point == 0:
                T.save(net.state_dict(), GN.WEIGHT_PATHS[model_name])
                data = {}
                with open(GN.JSON_PATHS[model_name], 'r') as f:
                    data = json.load(f)
                with open(GN.JSON_PATHS[model_name], 'w') as f:
                    data['all_loss'].append(all_loss)
                    json.dump(data, f)
                    all_loss = []
                print('checkpoint')
                con = input('continue?')
                if con == 'n':
                    T.cuda.empty_cache()
                    exit()
