import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


PATHS = {
    'STEM': 'D:/projects/GestureRecognition/src/config/InceptionResnetConfigs/Stem.cfg',
    'RESNET-A': 'D:/projects/GestureRecognition/src/config/InceptionResnetConfigs/ResnetModule-A.cfg',
    'REDUCTION-A': 'D:/projects/GestureRecognition/src/config/InceptionResnetConfigs/ReductionModule-A.cfg',
    'RESNET-B': 'D:/projects/GestureRecognition/src/config/InceptionResnetConfigs/ResnetModule-B.cfg',
    'REDUCTION-B': 'D:/projects/GestureRecognition/src/config/InceptionResnetConfigs/ReductionModule-B.cfg',
    'RESNET-C': 'D:/projects/GestureRecognition/src/config/InceptionResnetConfigs/ResnetModule-C.cfg',
    'REDUCTION-C': 'D:/projects/GestureRecognition/src/config/InceptionResnetConfigs/ReductionModule-C.cfg',
    'SCHEMA': 'D:/projects/GestureRecognition/src/config/InceptionResnetConfigs/Schema.cfg',
    'SCHEMA-2': 'D:/projects/GestureRecognition/src/config/InceptionResnetConfigs/Schema-2.cfg',
    'HPYOLO': 'D:/projects/GestureRecognition/src/config/handposeyolo.cfg',
    'STRIDEYOLO': 'D:/projects/GestureRecognition/src/config/handposestride.cfg',
    'MASKCNN': 'D:/projects/GestureRecognition/src/config/segmentation.cfg',
    'POSEMACHINE': 'D:/projects/GestureRecognition/src/config/HandPoseMachine.cfg',
    'HANDDETECTOR': 'D:/projects/GestureRecognition/src/config/hand_detector.cfg'
}

JSON_PATHS = {
    'HANDDETECTOR': 'D:/projects/GestureRecognition/src/Model/hand_bbox_data.json',
    'HPYOLO': 'D:/projects/GestureRecognition/src/Model/hand_pose_data_yolo.json',
    'STRIDEYOLO': 'D:/projects/GestureRecognition/src/Model/hand_pose_data_strideyolo.json',
    'MASKCNN': 'D:/projects/GestureRecognition/src/Model/hand_mask_data.json',
    'POSEMACHINE': 'D:/projects/GestureRecognition/src/Model/hand_pose_machine.json'
}

WEIGHT_PATHS = {
    'HPYOLO': 'D:/projects/GestureRecognition/src/Model/hand_pose_model_yolo.pth',
    'STRIDEYOLO': 'D:/projects/GestureRecognition/src/Model/hand_pose_data_stsrideyolo.pth',
    'MASKCNN': 'D:/projects/GestureRecognition/src/Model/hand_mask_model.pth',
    'POSEMACHINE': 'D:/projects/GestureRecognition/src/Model/hand_pose_machine_3.pth',
    'HANDDETECTOR': 'D:/projects/GestureRecognition/src/Model/hand_bbox_model_2.pth'
}


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


def parse_cfg(cfgfile, device):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    blocks = blocks
    module_list = nn.ModuleList()
    filters = int(blocks[0]['input_channels'])
    output_filters = []

    for index, block in enumerate(blocks):
        module = nn.Sequential()
        keys = block.keys()
        block_type = block['type']
        if block_type == 'conv':
            activation = block['activation']
            filters = int(block['filters'])
            kernel_size = tuple(map(int, block['size'].split(',')))
            kernel_size = kernel_size[0] if len(
                kernel_size) == 1 else kernel_size
            repeat = int(block['repeat']) if 'repeat' in keys else 1
            padding = int(block['pad']) if 'pad' in keys else 0
            stride = int(block['stride']) if 'stride' in keys else 1
            from_ = int(block['from']) if 'from' in keys else -1
            prev_filters = output_filters[index + from_]
            batch_norm = int(block['batchnorm']) if 'batchnorm' in keys else 1
            bias = bool(batch_norm)
            if padding:
                if type(kernel_size) == int:
                    pad = kernel_size // 2
                else:
                    pad = tuple([k // 2 for k in kernel_size])
            else:
                pad = 0
            for _ in range(repeat):
                conv = nn.Conv2d(prev_filters, filters,
                                kernel_size, stride, pad, bias=bias)
                module.add_module('conv_{0}_{1}'.format(index, _), conv)
                prev_filters = filters
                if batch_norm:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module('batch_norm_{0}_{1}'.format(index, _), bn)

                if activation == 'leaky':
                    activn = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module('relu_{0}_{1}'.format(index, _), activn)

            # if activation == 'linear':
            #     activn = nn.functional.linear()
            #     module.add_module('linear_{0}'.format(index), activn)
        elif block_type == 'maxpool':
            padding = int(block['pad']) if 'pad' in keys else 0
            kernel_size = tuple(map(int, block['size'].split(',')))
            kernel_size = kernel_size[0] if len(
                kernel_size) == 1 else kernel_size
            if padding:
                if type(kernel_size) == int:
                    pad = kernel_size // 2
                else:
                    pad = tuple([k // 2 for k in kernel_size])
            else:
                pad = 0
            mp = nn.MaxPool2d(kernel_size, int(block['stride']), pad)
            module.add_module('max_pool_2d_{}'.format(index), mp)
        elif block_type == 'concat':
            cc = EmptyLayer()
            module.add_module('concat_{}'.format(index), cc)
            filters = sum([output_filters[index + int(e)]
                           for e in block['layers'].split(',')])
        elif block_type == 'shortcut':
            if 'activation' in keys:
                if block['activation'] == 'leaky':
                    module.add_module('leaky_{}'.format(index), nn.LeakyReLU(0.1, inplace=True))
            else:
                module.add_module('shortcut_{}'.format(index), EmptyLayer())
        elif block_type == 'net':
            # re = nn.ReLU()
            # module.add_module('net_{}'.format(index), re)
            module.add_module('net_{}'.format(index), EmptyLayer())
        elif block_type == 'relu':
            re = nn.ReLU(inplace=True)
            module.add_module('relu_{}'.format(index), EmptyLayer())
        elif block_type in PATHS.keys():
            for i in range(int(block['repeat'])):
                module.add_module(''.join([block_type, '_', str(i + 1)]), GenericNet(PATHS[block_type], device).to(device))
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return module_list, blocks


class GenericNet(nn.Module):
    def __init__(self, cfg_path, device):
        super(GenericNet, self).__init__()
        self.module_list, self.blocks = parse_cfg(cfg_path, device)

    def forward(self, x, CUDA=True):
        outputs = {}
        returns = []
        for i, block in enumerate(self.blocks, 0):
            module_type = (block['type'])
            from_ = i + (int(block['from']) if 'from' in block.keys() else -1)
            if module_type == 'conv':
                x = self.module_list[i](outputs[from_])
            elif module_type == 'maxpool':
                x = self.module_list[i](outputs[from_])
            elif module_type == 'shortcut':
                layers = [*map(int, block['layers'].split(','))]
                for e in layers:
                    x = x + outputs[i + e]
                x = self.module_list[i](x)
            elif module_type == 'concat':
                layers = tuple([outputs[i + e]
                                for e in tuple(map(int, block['layers'].split(',')))])
                x = T.cat(layers, 1)
            elif module_type == 'relu':
                x = self.module_list[i](x)
            elif module_type in PATHS.keys():
                x = self.module_list[i](x)
            # elif module_type == 'net':
            #     x = self.module_list[i](x)
            outputs[i] = x
            # print(x[:,:,:,:].max())
            if 'output' in block.keys():
                returns.append(x)
        return returns
