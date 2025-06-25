import pickle
from collections import OrderedDict
from datasets.cifar10 import CIFAR10
from datasets.transforms import cifar_trans_test, cifar_trans_train
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.sampler as sampler
from train.model.performance import EpochPerformance
from utils.log import change_log_location, log_print
from utils.path_name import create_path_name
import os


class LabelGenerator(nn.Module):
    def __init__(self, psi, device):
        super(LabelGenerator, self).__init__()
        """
            label-generation network:
            takes the input and generates auxiliary labels with masked softmax for an auxiliary task.
        """
        filter = [64, 128, 256, 512, 512]
        self.class_nb = psi
        self.device = device

        # define convolution block in VGG-16
        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        # define fc-layers in VGG-16 (output auxiliary classes \sum_i\psi[i])
        self.classifier = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], int(np.sum(self.class_nb))),
        )

        # another head to output 0-1 value s.t. shape needs to be (1, batchsize)
        self.weight_head = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], 1),
            nn.Sigmoid()
        )

        # apply weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, in_channel, out_channel, index):
        if index < 3:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        return conv_block

    # define masked softmax
    def mask_softmax(self, x, mask, dim=1):
        logits = torch.exp(x) * mask / torch.sum(torch.exp(x) * mask, dim=dim, keepdim=True)
        return logits

    def forward(self, x, y):
        g_block1 = self.block1(x)
        g_block2 = self.block2(g_block1)
        g_block3 = self.block3(g_block2)
        g_block4 = self.block4(g_block3)
        g_block5 = self.block5(g_block4)

        # build a binary mask by psi, we add epsilon=1e-8 to avoid nans
        index = torch.zeros([len(self.class_nb), np.sum(self.class_nb)]) + 1e-8
        index = index.to(self.device)
        for i in range(len(self.class_nb)):
            index[i, int(np.sum(self.class_nb[:i])):np.sum(self.class_nb[:i+1])] = 1
        mask = index[y].to(self.device)

        predict = self.classifier(g_block5.view(g_block5.size(0), -1))
        label_pred = self.mask_softmax(predict, mask, dim=1)

        weight = self.weight_head(g_block5.view(g_block5.size(0), -1))

        return label_pred, weight


class SimplifiedVGG16(nn.Module):
    def __init__(self, num_primary_classes, device):
        super(SimplifiedVGG16, self).__init__()
        """
            multi-task network:
            takes the input and predicts primary and auxiliary labels (same network structure as in human)
        """
        filter = [64, 128, 256, 512, 512]
        self.device = device

        # define convolution block in VGG-16
        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        # primary task prediction
        self.classifier = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], num_primary_classes),
            nn.Softmax(dim=1)
        )

        # apply weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, in_channel, out_channel, index):
        if index < 3:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        return conv_block

    def forward(self, x):
        """
            if no weights given, use the direct training strategy and update network paramters
            else retain the computational graph which will be used in second-derivative step
        """

        g_block1 = self.block1(x)
        g_block2 = self.block2(g_block1)
        g_block3 = self.block3(g_block2)
        g_block4 = self.block4(g_block3)
        g_block5 = self.block5(g_block4)

        t1_pred = self.classifier(g_block5.view(g_block5.size(0), -1))

        return t1_pred

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.state_dict(), file_path)

    @classmethod
    def load(cls, file_path: str,
             num_primary_classes: int,
              device: torch.device = torch.device("cpu")):
        model = cls(num_primary_classes=num_primary_classes, device=device)
        model.load_state_dict(torch.load(file_path, map_location=device))
        model.to(device)
        model.eval()       # or `.train()` if you plan to keep training
        return model

