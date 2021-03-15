from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import time

def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """

    file = open(cfgfile, 'r')
    # store the lines in a list
    lines = file.read().split('\n')
    # get read of the empty lines
    lines = [x for x in lines if len(x) > 0]
    # get rid of comments
    lines = [x for x in lines if x[0] != '#']
    # get rid of fringe whitespaces
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            # If block is not empty, implies it is storing values of previous block.
            if len(block) != 0:
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


blocks = parse_cfg('D:/projects/GestureRecognition/src/config/yolov3-tiny.cfg')


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    # Captures the information about the input and pre-processing
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # check the type of block
        # create a new module for the block
        # append to module_list

        # If it's a convolutional layer
        if (x["type"] == "convolutional"):
            # Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters,
                             kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

            # If it's an upsampling layer
            # We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        # If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            # Start  of a route
            start = int(x["layers"][0])
            # end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index +
                                         start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        elif x['type'] == 'maxpool':
            mp = nn.MaxPool2d(int(x['size']), int(x['stride']))
            module.add_module('max_pool_2d_{}'.format(index), mp)
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    # batch_size = prediction.size(0)
    # stride = inp_dim // prediction.size(2)
    # grid_x, grid_y = prediction.size(2), prediction.size(3)
    # bbox_attrs = 5 + num_classes
    # num_anchors = len(anchors)

    # prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_x * grid_y)
    # prediction = prediction.transpose(1, 2)
    # prediction = prediction.view(batch_size, grid_x * grid_y * num_anchors, bbox_attrs)
    # anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    # torch.sigmoid_(prediction[:, :, :, :])
    # prediction *= 100
    # prediction = prediction.contiguous()
    return prediction

    for i in [0, 1, 4]:
        prediction[:, :, i] = torch.sigmoid(prediction[:, :, i])
    # Add the center offsets
    a, b = np.meshgrid(np.arange(grid_x), np.arange(grid_y))
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, num_anchors).view(-1, 2).unsqueeze(0)
    if CUDA:
        x_y_offset = x_y_offset.cuda()
        prediction = prediction.cuda()
    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_x*grid_y, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4])*anchors

    prediction[:, :, :4] *= stride
    return prediction


class TinyYolo(nn.Module):
    def __init__(self, blocks):
        super(TinyYolo, self).__init__()
        self.blocks = blocks
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA = True):
        modules = self.blocks[1:]
        outputs = {}  # We cache the outputs for the route layer
        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample" or module_type == 'maxpool':
                x = self.module_list[i](x)
                # print(i, x.shape, 'after', module_type)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                # print(i, x.shape, 'route from', *layers)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
                # print(i, x.shape, 'shortcut from', from_)

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform
                # x = x.data
                # print(i, x.shape, 'after yolo')
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                # print(x.shape)
                # print('after yolo ', x.shape)
                return x
                # if not write:  # if no collector has been intialised.
                #     detections = x
                #     write = 1
                # else:
                #     detections = torch.cat((detections, x), 1)

            outputs[i] = x

        # return detections

# cnt = 0
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net = TinyYolo(blocks).to(device)
# net.eval()
# ten = torch.randn(1, 3, 416, 312).to(device)
# ts = None
# while cnt < 215:
#     if ts == None and cnt == 1:
#         ts = time.time()
#         print('starting')
#     preds = net(ten, True)
#     cnt += 1
# print(time.time() - ts)
# print(preds.shape)
# print(preds)
# res = utils.write_results(preds, 0.5, 80)
# print(res.shape)
# print(res[:, 7])