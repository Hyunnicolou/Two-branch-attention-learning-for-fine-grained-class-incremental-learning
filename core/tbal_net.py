#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/5 20:13
# @Author : Jiaqi Guo
# @Site : 
# @File : tbal_net.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from core.cbam import CBAM
import torch.functional as F

input_size = 448
N_list = [2, 3, 2]
stride = 32

proposalN = sum(N_list)  # proposal window num
window_side = [128, 192, 256]
iou_threshs = [0.15, 0.15, 0.15]
ratios = [[4, 4], [3, 5], [5, 3],
        [6, 6], [5, 7], [7, 5],
        [8, 8], [6, 10], [10, 6], [7, 9], [9, 7], [7, 10], [10, 7]]


def compute_window_nums(ratios, stride, input_size):
    size = input_size / stride
    window_nums = []

    for _, ratio in enumerate(ratios):
        window_nums.append(int((size - ratio[0]) + 1) * int((size - ratio[1]) + 1))

    return window_nums

def ComputeCoordinate(image_size, stride, indice, ratio):
    size = int(image_size / stride)
    column_window_num = (size - ratio[1]) + 1
    x_indice = indice // column_window_num
    y_indice = indice % column_window_num
    x_lefttop = x_indice * stride - 1
    y_lefttop = y_indice * stride - 1
    x_rightlow = x_lefttop + ratio[0] * stride
    y_rightlow = y_lefttop + ratio[1] * stride
    # for image
    if x_lefttop < 0:
        x_lefttop = 0
    if y_lefttop < 0:
        y_lefttop = 0
    coordinate = np.array((x_lefttop, y_lefttop, x_rightlow, y_rightlow)).reshape(1, 4)

    return coordinate


def indices2coordinates(indices, stride, image_size, ratio):
    batch, _ = indices.shape
    coordinates = []

    for j, indice in enumerate(indices):
        coordinates.append(ComputeCoordinate(image_size, stride, indice, ratio))

    coordinates = np.array(coordinates).reshape(batch, 4).astype(int)       # [N, 4]
    return coordinates


window_nums = compute_window_nums(ratios, stride, input_size) # 在输入大小为input_size的图像上，以步长为stride，当长宽比为ratio时的窗口个数，输出为每个ratio的窗口数
indices_ndarrays = [np.arange(0, window_num).reshape(-1,1) for window_num in window_nums]
coordinates = [indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)] # 每个window在image上的坐标
coordinates_cat = np.concatenate(coordinates, 0)
window_milestones = [sum(window_nums[:i+1]) for i in range(len(window_nums))]

window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:6]), sum(window_nums[6:])]

def nms(scores_np, proposalN, iou_threshs, coordinates):
    if not (type(scores_np).__module__ == 'numpy' and len(scores_np.shape) == 2 and scores_np.shape[1] == 1):
        raise TypeError('score_np is not right')

    windows_num = scores_np.shape[0]
    indices_coordinates = np.concatenate((scores_np, coordinates), 1)

    indices = np.argsort(indices_coordinates[:, 0])
    indices_coordinates = np.concatenate((indices_coordinates, np.arange(0,windows_num).reshape(windows_num,1)), 1)[indices]                  #[339,6]
    indices_results = []

    res = indices_coordinates

    while res.any():
        indice_coordinates = res[-1]
        indices_results.append(indice_coordinates[5])

        if len(indices_results) == proposalN:
            return np.array(indices_results).reshape(1, proposalN).astype(np.int)
        res = res[:-1]

        # Exclude anchor boxes with selected anchor box whose iou is greater than the threshold
        start_max = np.maximum(res[:, 1:3], indice_coordinates[1:3])
        end_min = np.minimum(res[:, 3:5], indice_coordinates[3:5])
        lengths = end_min - start_max + 1
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1] + 1) * (res[:, 4] - res[:, 2] + 1) +
                                      (indice_coordinates[3] - indice_coordinates[1] + 1) *
                                      (indice_coordinates[4] - indice_coordinates[2] + 1) - intersec_map)
        res = res[iou_map_cur <= iou_threshs]

    while len(indices_results) != proposalN:
        indices_results.append(indice_coordinates[5])

    return np.array(indices_results).reshape(1, -1).astype(np.int)

class APPM(nn.Module):
    def __init__(self):
        super(APPM, self).__init__()
        self.avgpools = [nn.AvgPool2d(ratios[i], 1) for i in range(len(ratios))]
        # self.avgpools = [nn.MaxPool2d(ratios[i], 1) for i in range(len(ratios))]

    def forward(self, proposalN, x, ratios, window_nums_sum, N_list, iou_threshs, DEVICE='cuda'):
        batch, channels, _, _ = x.size()
        avgs = [self.avgpools[i](x) for i in range(len(ratios))]  # 使用不同size的kernel对输入进行池化操作，得到一个list
                                                                # list中每个元素为
        # feature map sum
        fm_sum = [torch.sum(avgs[i], dim=1) for i in range(len(ratios))] # 在通道维度进行求和操作,得到的数据size为（B,1,h,w）

        all_scores = torch.cat([fm_sum[i].view(batch, -1, 1) for i in range(len(ratios))], dim=1)
        # 在每个ratio的情况下，都会得到这种情况下的score分布
        windows_scores_np = all_scores.data.cpu().numpy() #
#        print('Windoes-scores_np shape ' + str(windows_scores_np.shape)) # (batchsize, 917, 1)
        window_scores = torch.from_numpy(windows_scores_np).to(DEVICE).reshape(batch, -1)
        '''
        ratios包含不同的窗口大小，以这些窗口
        '''
        # nms
        proposalN_indices = []
        for i, scores in enumerate(windows_scores_np): # 这个地方是对batch里所有图像结果的遍历
#            print('scores shapes ' + str(scores.shape)) # [917,1]
            indices_results = []
            for j in range(len(window_nums_sum)-1):
                indices_results.append(nms(scores[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])], proposalN=N_list[j], iou_threshs=iou_threshs[j],
                                           coordinates=coordinates_cat[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])]) + sum(window_nums_sum[:j+1]))
            # indices_results.reverse()
            proposalN_indices.append(np.concatenate(indices_results, 1)) # reverse

        proposalN_indices = np.array(proposalN_indices).reshape(batch, proposalN)
        proposalN_indices = torch.from_numpy(proposalN_indices).to(DEVICE)
        proposalN_windows_scores = torch.cat(
            [torch.index_select(all_score, dim=0, index=proposalN_indices[i].long()) for i, all_score in enumerate(all_scores)], 0).reshape(
            batch, proposalN)

        return proposalN_indices, proposalN_windows_scores, window_scores


class TBAL_Net(nn.Module):
    def __init__(self, proposalN = proposalN, num_classes = 200):
        self.proposalN = proposalN
        self.num_classes = num_classes

        resnet50 = models.resnet50(pretrained=True)
        self.reg_conv1 = resnet50.conv1
        self.reg_bn1 = resnet50.bn1
        self.reg_relu = resnet50.relu
        self.reg_maxpool = resnet50.maxpool
        self.reg_layer1 = resnet50.layer1
        self.reg_layer2 = resnet50.layer2
        self.reg_layer3 = resnet50.layer3
        self.reg_layer4 = resnet50.layer4
        self.reg_avgpool = nn.AdaptiveAvgPool2d(1)
        del resnet50

        self.fc = nn.Linear(2048, self.num_classes)

        self.cbam_1 = CBAM(gate_channels=2048)
        self.cbam_2 = CBAM(gate_channels=2048)

        self.APLM = APPM()

    def backbone_forward(self, inputs):
        conv1 = self.reg_conv1(inputs)
        conv1 = self.reg_bn1(conv1)
        conv1 = self.reg_relu(conv1)

        conv1 = self.reg_maxpool(conv1)
        layer1 = self.reg_layer1(conv1)
        layer2 = self.reg_layer2(layer1)
        layer3 = self.reg_layer3(layer2)
        layer4_b = self.reg_layer4[:2](layer3)
        layer4_b = self.cbam_1(layer4_b)
        layer4 = self.reg_layer4[2](layer4_b)
        layer4 = self.cbam_2(layer4)
        outputs = self.reg_avgpool(layer4)

        return layer4_b, layer4, outputs

    def forward(self, inputs, DEVICE='cuda'):
        batch_size, channels, H, W = inputs.size()

        _, layer4, outputs = self.backbone_forward(inputs)
        backbone_logits = self.fc(outputs)

        proposalN_indices, proposalN_windows_scores, window_scores \
            = self.APLM(self.proposalN, layer4.detach(), ratios, window_nums_sum, N_list, iou_threshs, DEVICE)

        window_imgs = torch.zeros([batch_size, self.proposalN, 3, 224, 224]).to(DEVICE)  # [N, 4, 3, 224, 224]

        for i in range(batch_size):
            for j in range(self.proposalN):
                [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[i, j]]
                window_imgs[i:i + 1, j] = F.interpolate(inputs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)],
                                                        size=(224, 224),
                                                        mode='bilinear',
                                                        align_corners=True)  # [N, 4, 3, 224, 224]

        window_imgs = window_imgs.reshape(batch_size * self.proposalN, 3, 224, 224)  # [N*4, 3, 224, 224]

        _, _, outputs = self.backbone_forward(window_imgs)
        local_logits = self.fc(outputs)

        return backbone_logits, local_logits, window_imgs










