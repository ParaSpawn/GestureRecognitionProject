import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import Nets.GenericNet as GenericModule
import Datasets.HandPoseDataset as DS
import numpy as np
import torch.optim as optim
import json
import os.path
import gc
import cv2
import torchvision.transforms as transforms


T.cuda.empty_cache()
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
print(T.cuda.get_device_name(device))
data_set = DS.HandPoseDatasetHPD(DS.PATHS['hpd_pose_data'], DS.PATHS['hpd_images_n'], [(58, 76), (39, 51), (29, 39)], device)


def gauss_2d(mu, sigma):
    x = random.gauss(mu, sigma)
    y = random.gauss(mu, sigma)
    return (x, y)


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
train_loader, test_loader = create_loaders(data_set, batch_size, 0.3, True, 69)

model_name = 'POSEMACHINE'
net = GenericModule.GenericNet(GenericModule.PATHS[model_name], device).to(device)
net.load_state_dict(T.load(GenericModule.WEIGHT_PATHS[model_name]))
net.train()

criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001)

floatTensor = T.cuda.FloatTensor

try:
    with open(GenericModule.JSON_PATHS[model_name], 'x') as f:
        with open(GenericModule.JSON_PATHS[model_name], 'w') as fi:
            data = {'all_loss': []}
            json.dump(data, fi)
except:
    print('file already exists')

all_loss = []
scales = [1.5, 1.0, 0.75]
with T.enable_grad():
    for epoch in range(2):
        for i, d in enumerate(train_loader, 0):
            images, heatmaps = d
            imgs, htmps = [], []
            for s, scale in enumerate(scales):
                imgs.append(T.zeros(batch_size, 3, int(300 * scale), int(300 * scale)))
                imgs[-1] = images[s][:, :, :].to(device)
                htmps.append(T.zeros(batch_size, 2, int(300 * scale), int(300 * scale)))
                htmps[-1] = heatmaps[s].to(device)
            c_loss = []
            for ind in range(len(imgs)):
                inp = imgs[ind].to(device)
                inp.requires_grad = True
                outputs = net(inp)
                cd_loss = []
                losses = []
                for t in outputs:
                    l = criterion(t, htmps[ind])
                    losses.append(l)
                    cd_loss.append(l.item())
                optimizer.zero_grad()
                loss = sum(losses)
                loss.backward()
                optimizer.step()
                c_loss.append(sum(cd_loss))
            print(i, '{:.4f}'.format(sum(c_loss)), '{0:.4f} {1:.4f} {2:.4f}'.format(*c_loss))
            all_loss.append(sum(c_loss))
            if i % 100 == 0 and i > 0:
                T.save(net.state_dict(), GenericModule.WEIGHT_PATHS[model_name])
                data = {}
                with open(GenericModule.JSON_PATHS[model_name], 'r') as f:
                    data = json.load(f)
                with open(GenericModule.JSON_PATHS[model_name], 'w') as f:
                    data['all_loss'].append(all_loss)
                    json.dump(data, f)
                    all_loss = []
                print('checkpoint')
                con = input('continue?')
                if con == 'n':
                    T.cuda.empty_cache()
                    exit()