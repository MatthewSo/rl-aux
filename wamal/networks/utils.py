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

def model_fit(x_pred, x_output, pri=True, num_output=3):
    if not pri:
        # generated auxiliary label is a soft-assignment vector (no need to change into one-hot vector)
        x_output_onehot = x_output
    else:
        # convert a single label into a one-hot vector
        x_output_onehot = torch.zeros((len(x_output), num_output)).to(self.device)
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

    # apply focal loss
    loss = x_output_onehot * (1 - x_pred)**2 * torch.log(x_pred + 1e-20)
    return torch.sum(-loss, dim=1)

def model_entropy(x_pred1):
    # compute entropy loss
    x_pred1 = torch.mean(x_pred1, dim=0)
    loss1 = x_pred1 * torch.log(x_pred1 + 1e-20)
    return torch.sum(loss1)