
import gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gym import spaces
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from tqdm import tqdm
import copy
import random
import sys
from stable_baselines3 import PPO
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from create_dataset import CIFAR100,ImbalancedDatasetWrapper
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import sys

from rl_text import f1_score_from_logits

def f1_score_from_logits(logits, labels):
    """
    Compute the F1 score from logits and labels for a multiclass classification task.

    Args:
    - logits (torch.Tensor): A tensor of shape (batch_size, num_classes) representing the model's raw predictions.
    - labels (torch.Tensor): A tensor of shape (batch_size,) containing the true class labels.

    Returns:
    - float: The macro-averaged F1 score.
    """
    preds = torch.argmax(logits, dim=1)

    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    f1 = f1_score(labels, preds, average='macro')

    return f1

def evaluate(model, data_loader, criterion, device, get_f1=False):
    model.eval()
    total_loss = 0
    correct = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            class_output, _ = model(inputs)
            loss = criterion(class_output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(class_output, 1)
            correct += (predicted == labels).sum().item()
            if get_f1:
                all_logits.append(class_output)
                all_labels.append(labels)

    accuracy = correct / len(data_loader.dataset)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    f1 = f1_score_from_logits(all_logits, all_labels)
    return total_loss / len(data_loader), accuracy, f1
