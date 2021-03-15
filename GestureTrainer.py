import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import Nets.GestureRNN as RNN
import Datasets.HandPoseDataset as DS
import numpy as np
import torch.optim as optim
import json
import os.path
import gc
import cv2
import torchvision.transforms as transforms
from PIL import Image


T.cuda.empty_cache()
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")



loc = 'D:/projects/GestureRecognition/Data/GestureDataset/1/'

imgs = []
to_ten = transforms.ToTensor()
for i in range(1, 6):
    imgs.append(to_ten(Image.open(loc + str(i) + '.bmp')).to(device).view(-1))

model_name = 'GESTURECLASSIFIER_1'
net = RNN.GestureRNN(10, 5, imgs[0].shape[0], device).to(device)
# net.load_state_dict(T.load(RNN.WEIGHT_PATHS[model_name]))
net.train()


try:
    with open(RNN.JSON_PATHS[model_name], 'x') as f:
        with open(RNN.JSON_PATHS[model_name], 'w') as fi:
            data = {'all_loss': []}
            json.dump(data, fi)
except:
    print('file already exists')


criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001)

target = T.zeros(10).to(device)
target[3] = 1

all_loss = []
with T.enable_grad():
    for i in range(201):
        outputs = net(imgs)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(i, '{:.10f}'.format(loss.item()))
        all_loss.append(loss.item())
        if i % 100 == 0 and i > 0:
            T.save(net.state_dict(), RNN.WEIGHT_PATHS[model_name])
            data = {}
            with open(RNN.JSON_PATHS[model_name], 'r') as f:
                data = json.load(f)
            with open(RNN.JSON_PATHS[model_name], 'w') as f:
                data['all_loss'].append(all_loss)
                json.dump(data, f)
                all_loss = []
            print('checkpoint')
            con = input('continue?')
            if con == 'n':
                T.cuda.empty_cache()
                exit()