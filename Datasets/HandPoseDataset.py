import torchvision
import torchvision.transforms as transforms
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv
import os
import operator
from PIL import Image
from PIL import ImageDraw
from collections import Counter
from scipy.stats import multivariate_normal
import random
import cv2

# device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

PATHS = {
    'frie_training_images': 'D:/projects/GestureRecognition/Data/FrieHandDataset/training/rgb/',
    'frie_training_pose_data': 'D:/projects/GestureRecognition/Data/FrieHandDataset/generated_data/pose_data.csv',
    'frie_training_masks': 'D:/projects/GestureRecognition/Data/FrieHandDataset/training/mask/',
    'hpd_images': 'D:/projects/GestureRecognition/Data/hand_pose_dataset/augmented_samples/',
    'hpd_bbox_data': 'D:/projects/GestureRecognition/Data/hand_pose_dataset/generated_data/allData/BBox_full.csv',
    'hpd_pose_data': 'D:/projects/GestureRecognition/Data/hand_pose_dataset/generated_data/allData/2Dpoints.csv',
    'hpd_images_n': 'D:/projects/GestureRecognition/Data/hand_pose_dataset/annotated_frames/'
}


class HandPoseDatasetFrie(T.utils.data.Dataset):
    def __init__(self, csv_pose_path, images_path, strides, device, transform=None):
        self.images_path = images_path
        # self.csv_bbox_data = pd.read_csv(csv_bbox_path)
        self.csv_pose_data = pd.read_csv(csv_pose_path)
        self.transform = transform
        self.device = device
        self.strides = strides
        self.patch_size = 25
        pos = np.dstack(np.mgrid[0:self.patch_size:1, 0:self.patch_size:1])
        rv = multivariate_normal(
            mean=[self.patch_size // 2, self.patch_size // 2], cov=self.patch_size)
        patch = T.tensor(rv.pdf(pos)).to(self.device)
        f = 5 / patch[:, :].max()
        patch[:, :] *= f
        self.patch = patch

    def __len__(self):
        return 130240 - 32560

    def get_grid_dim(self):
        return [self.img_size[1] * self.scaling_factor // self.stride, self.img_size[0] * self.scaling_factor // self.stride]

    def __getitem__(self, i):
        img_name = self.images_path + '{:08d}'.format(i + 32560) + '.jpg'
        image = Image.open(img_name)
        pose_data = np.array(self.csv_pose_data.iloc[i % 32560, :])
        pose = []
        for k in [0, 3]:
            for i in range(1, 6):
                for j in range(2):
                    pose.append(pose_data[(4 * i - k) * 2 + j])
        pose = pose[:10]
        minx = min(pose[0::2]) - 60
        maxx = max(pose[0::2]) + 60
        miny = min(pose[1::2]) - 60
        maxy = max(pose[1::2]) + 60
        box = [minx, miny, maxx - minx, maxy - miny]
        side_length = max(box[2:])
        corner = box[:2]
        if box[2] > box[3]:
            corner[1] -= (box[2] - box[3]) / 2
        else:
            corner[0] -= (box[3] - box[2]) / 2
        box[0] = corner[0]
        box[1] = corner[1]
        box[2] = box[3] = side_length
        size = 300
        for i in range(0, len(pose), 2):
            pose[i] -= corner[0]
            pose[i] *= size / side_length
            pose[i + 1] -= corner[1]
            pose[i + 1] *= size / side_length
        image = transforms.functional.resized_crop(image, int(corner[1]), int(
            corner[0]), int(side_length), int(side_length), (size, size))
        scales = [1.5, 1.0, 0.75]
        images = [None for _ in range(len(scales))]
        image = np.array(image)
        for i, scale in enumerate(scales):
            images[i] = cv2.resize(
                image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            images[i] = transforms.ToTensor()(images[i]).to(self.device)

        pose = T.tensor([[
            pose[i],
            pose[i + 1]
        ] for i in range(0, len(pose), 2)])
        heatmaps = [None for _ in range(len(scales))]
        for j, stride in enumerate(self.strides):
            hs = T.zeros(5, stride, stride)
            for i, xy in enumerate(pose):
                bg = T.zeros(300, 300)
                bg_x_l = int(max(0, xy[0] - self.patch_size // 2))
                bg_x_r = int(min(xy[0] + self.patch_size // 2, 300))
                bg_y_b = int(max(0, xy[1] - self.patch_size // 2))
                bg_y_t = int(min(xy[1] + self.patch_size // 2, 300))
                patch_x_l = 0 - min(0, xy[0] - self.patch_size // 2)
                # self.patch_size + min(300 - xy[0] + self.patch_size // 2, 0)
                patch_x_r = patch_x_l + (bg_x_r - bg_x_l)
                patch_y_b = 0 - min(0, xy[1] - self.patch_size // 2)
                # self.patch_size + min(300 - xy[1] + self.patch_size // 2, 0)
                patch_y_t = patch_y_b + (bg_y_t - bg_y_b)
                bg[bg_x_l:bg_x_r, bg_y_b:bg_y_t] = self.patch[patch_x_l:patch_x_r,
                                                              patch_y_b:patch_y_t]
                bg = T.transpose(bg, 1, 0)
                hs[i] = T.tensor(cv2.resize(np.array(
                    bg.cpu()), (0, 0), fx=stride/300, fy=stride/300, interpolation=cv2.INTER_CUBIC))
            heatmaps[j] = hs
        # return image, pose
        return images, heatmaps


class HandPoseDatasetHPD(T.utils.data.Dataset):
    def __init__(self, csv_pose_path, images_path, strides, device, transform=None):
        self.images_path = images_path
        # self.csv_bbox_data = pd.read_csv(csv_bbox_path)
        self.csv_pose_data = pd.read_csv(csv_pose_path)
        self.transform = transform
        self.device = device
        self.strides = strides
        self.patch_size = 20
        pos = np.dstack(np.mgrid[0:self.patch_size:1, 0:self.patch_size:1])
        rv = multivariate_normal(
            mean=[self.patch_size // 2, self.patch_size // 2], cov=self.patch_size)
        patch = T.tensor(rv.pdf(pos)).to(self.device)
        f = 10 / patch[:, :].max()
        patch[:, :] *= f
        self.patch = patch

    def __len__(self):
        return len(self.csv_pose_data)

    def get_grid_dim(self):
        return [self.img_size[1] * self.scaling_factor // self.stride, self.img_size[0] * self.scaling_factor // self.stride]

    def __getitem__(self, i):
        img_name = self.images_path + self.csv_pose_data.iloc[i, 0]
        image = Image.open(img_name)
        pose = list(np.array(self.csv_pose_data.iloc[i, 1:-2]))
        # minx = min(pose[0::2]) - 60
        # maxx = max(pose[0::2]) + 60
        # miny = min(pose[1::2]) - 60
        # maxy = max(pose[1::2]) + 60
        # box = [minx, miny, maxx - minx, maxy - miny]
        # side_length = max(box[2:])
        # corner = box[:2]
        # if box[2] > box[3]:
        #     corner[1] -= (box[2] - box[3]) / 2
        # else:
        #     corner[0] -= (box[3] - box[2]) / 2
        # box[0] = corner[0]
        # box[1] = corner[1]
        # box[2] = box[3] = side_length
        size = (300, int(300 * 640 / 480))
        for i in range(0, len(pose), 2):
            pose[i] *= 300 / 480
            pose[i + 1] *= 300 / 480
        #     pose[i] -= corner[0]
            # pose[i] *= size / side_length
        #     pose[i + 1] -= corner[1]
            # pose[i + 1] *= size / side_length
        image = transforms.Resize(size)(image)
        # image = transforms.functional.resized_crop(image, int(corner[1]), int(
        #     corner[0]), int(side_length), int(side_length), (size, size))
        scales = [1.5, 1.0, 0.75]
        images = [None for _ in range(len(scales))]
        image = np.array(image)
        for i, scale in enumerate(scales):
            images[i] = cv2.resize(
                image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            images[i] = transforms.ToTensor()(images[i]).to(self.device)

        pose = T.tensor([[
            pose[i],
            pose[i + 1]
        ] for i in range(0, len(pose), 2)])

        heatmaps = [None for _ in range(len(scales))]
        for j, stride in enumerate(self.strides):
            hs = T.zeros(2, stride[0], stride[1]).to(self.device)
            for l in range(2):
                bg = T.zeros(size[1], size[0]).to(self.device)
                for k, xy in enumerate(pose[l * 5:((l + 1) * 5)]):
                    bg_x_l = int(max(0, xy[0] - self.patch_size // 2))
                    bg_x_r = int(min(xy[0] + self.patch_size // 2, size[0]))
                    bg_y_b = int(max(0, xy[1] - self.patch_size // 2))
                    bg_y_t = int(min(xy[1] + self.patch_size // 2, size[1]))
                    patch_x_l = 0 - min(0, xy[0] - self.patch_size // 2)
                    # self.patch_size + min(300 - xy[0] + self.patch_size // 2, 0)
                    patch_x_r = patch_x_l + (bg_x_r - bg_x_l)
                    patch_y_b = 0 - min(0, xy[1] - self.patch_size // 2)
                    # self.patch_size + min(300 - xy[1] + self.patch_size // 2, 0)
                    patch_y_t = patch_y_b + (bg_y_t - bg_y_b)
                    try:
                        bg[bg_x_l:bg_x_r, bg_y_b:bg_y_t] = bg[bg_x_l:bg_x_r, bg_y_b:bg_y_t] + \
                            self.patch[patch_x_l:patch_x_r,
                                       patch_y_b:patch_y_t]
                    except:
                        print('error patching')
                bg = T.transpose(bg, 1, 0)
                hs[l] = T.tensor(cv2.resize(np.array(
                    bg.cpu()), (0, 0), fx=stride[1]/size[1], fy=stride[0]/size[0], interpolation=cv2.INTER_CUBIC))
            heatmaps[j] = hs
        # return image, pose
        return images, heatmaps

# def testing2():
#     ds = HandMaskDataset(PATHS['frie_training_images'], PATHS['frie_training_masks'], T.device("cuda:0" if T.cuda.is_available() else "cpu"))
#     img, msk = ds[0]
#     return img, msk
# def testing():
#     import matplotlib.pyplot as plt
#     import random

#     device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
#     hpd = HandPoseDatasetHPD(PATHS['hpd_pose_data'], PATHS['hpd_images'], device, stride=17)
#     image, pose = hpd[random.randrange(len(hpd))]
#     x = []
#     y = []
#     for i in range(len(pose)):
#         x.append(int(pose[i][0] * 17 + 17 * pose[i][3] / 100))
#         y.append(int(pose[i][1] * 17 + 17 * pose[i][4] / 100))
#     plt.imshow(image)
#     plt.scatter(x, y)


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
        size = image.size
        coord = T.tensor(
            [(box[0] + box[2] / 2) / size[0],
             (box[1] + box[3] / 2) / size[1],
             max(box[2], box[3]) / size[1]]
        ).to(self.device)
        images = []
        for scale in self.scales:
            t = transforms.Compose([transforms.Resize(
                (int(scale * size[1]), int(scale * size[0]))), transforms.ToTensor()])
            images.append(t(image).to(self.device))
        return images, coord


class GestureDataset(T.utils.data.Dataset):
    def __init__(self, root_path, device):
        self.root_path = root_path
        self.device = device

    def __len__(self):
        return 1

    def __getitem__(self, i):
        loc = self.root_path + str(i)
        ret = []
        for j in range(5):
            ret.append(transforms.ToTensor()(
                Image.open(loc + '/' + str(j))).to(self.device))
        return ret
