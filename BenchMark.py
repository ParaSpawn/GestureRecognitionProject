import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import GenericNet

ten = T.randn(1, 3, 400, 400).to(GenericNet.device)
ts = None
cnt = 0
while cnt < 40:
    if ts == None and cnt == 1:
        ts = time.time()
        print('starting')
    preds = GenericNet.net(ten, True)
    cnt += 1
print(time.time() - ts)