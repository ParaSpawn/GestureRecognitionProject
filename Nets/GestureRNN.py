import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


JSON_PATHS = {
    'GESTURECLASSIFIER_1': 'D:/projects/GestureRecognition/src/Model/gesture_classifier_1.json'
}

WEIGHT_PATHS = {
    'GESTURECLASSIFIER_1': 'D:/projects/GestureRecognition/src/Model/gesture_classifier_1.pth'
}


class RNNModule(nn.Module):
    def __init__(self, cell_count, out_going=100):
        super(RNNModule, self).__init__()
        self.x_mod = nn.Sequential()
        self.x_mod.add_module('fc_1', nn.Linear(cell_count, 1000, bias=True))
        self.x_mod.add_module('leaky_1', nn.LeakyReLU(0.05, True))
        self.x_mod.add_module('fc_2', nn.Linear(1000, 1000, bias=True))
        self.x_mod.add_module('leaky_2', nn.LeakyReLU(0.05, True))
        self.x_mod.add_module('fc_3', nn.Linear(1000, out_going, bias=True))
        self.x_mod.add_module('leaky_3', nn.LeakyReLU(0.05, True))
        self.a_mod = nn.Linear(100, 100, bias=True)
        self.act = nn.LeakyReLU(0.05, True)

    def forward(self, x, a):
        x = self.x_mod(x) + self.a_mod(a)
        x = self.act(x)
        return x


class GestureRNN(nn.Module):
    def __init__(self, gesture_count, frame_count, cell_count, device, out_going=100):
        super(GestureRNN, self).__init__()
        self.gesture_count = gesture_count
        self.frame_count = frame_count
        self.seq = nn.ModuleList()
        for i in range(frame_count):
            self.seq.add_module('rnn_{}'.format(i), RNNModule(cell_count, out_going))
        self.final = nn.Linear(out_going, gesture_count, bias=True)
        self.out_going = out_going
        self.device = device

    
    def forward(self, input_list):
        a = T.zeros(self.out_going).to(self.device)
        for i, m in enumerate(self.seq):
            a = m(input_list[i], a)
        ret = self.final(a)
        return ret
