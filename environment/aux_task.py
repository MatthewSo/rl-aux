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

from rl_text import SeededSubsetRandomSampler
from utils.evaluate import evaluate, f1_score_from_logits
from utils.masked_softmax import create_mask_from_labels, mask_softmax
from utils.vars import softmax


class AuxTaskEnv(gym.Env):
    def __init__(self, train_dataset, device,model,criterion, optimizer_func, scheduler_func,batch_size=64,aux_dim=50,verbose=False,aux_weight = 1):
        super(AuxTaskEnv, self).__init__()
        self.batch_size=batch_size
        self.train_dataset=train_dataset
        self.cannonical_model=copy.deepcopy(model)
        self.criterion = criterion
        self.verbose=verbose
        self.aux_weight=aux_weight

        self.optimizer_func=optimizer_func
        self.scheduler_func=scheduler_func

        # optimizer and its reload state
        self.optimizer_reload_state=None
        self.optimizer = None
        self.scheduler_reload_state = None
        self.scheduler = None

        # Initialize environment variables
        self.device = device

        # initialize dataloader with seed
        self.randomize_seed()
        self.reset_data_loader(self.seed)
        self.current_batch = None
        self.current_batch_index = 0
        self.current_batch_aux_labels = []

        # create endless sampler for reward sampling
        sampler = RandomSampler(train_dataset, replacement=True,num_samples=sys.maxsize )
        self.reward_sampler = iter( DataLoader(train_dataset, batch_size=256, sampler=sampler))
        self.reward_mode = True # toggle reward calculation when not training

        # Define action and observation space
        image_obs = spaces.Box(low=0, high=1, shape=(3, 32, 32), dtype=np.float32)
        label_obs = spaces.Box(low=0, high=100, shape=(100,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "image": image_obs,
            "array": label_obs,
        })

        print(self.observation_space)
        self.action_space = spaces.Box(low=-1, high=1, shape=(batch_size, aux_dim), dtype=np.float32)

        # Model for the main classification task (reset this each episode)
        self.model=copy.deepcopy(self.cannonical_model).to(self.device)

        # step counter and return counter
        self.count=0
        self.return_ = 0

    def randomize_seed(self):
        self.seed = random.randint(0, 2**32 - 1)

    def reset_data_loader(self,seed):
        # Define the indices for the dataset
        indices = list(range(len(self.train_dataset)))
        sampler = SeededSubsetRandomSampler(indices, seed=seed)

        self.train_loader  = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler)

    def get_stats(self):
        print("return:",self.return_)
        print("len",self.count)

    def update(self):
        self.scheduler.step()
        self.cannonical_model=copy.deepcopy(self.model)
        self.optimizer_reload_state=copy.deepcopy( self.optimizer.state_dict())
        self.scheduler_reload_state=copy.deepcopy( self.scheduler.state_dict())
        self.randomize_seed()

    def evaluate(self,test_loader):
        # Evaluate the network on train and test sets
        if self.model is not None:
            train_accuracy = evaluate(self.model, self.train_loader,self.criterion, self.device)
            test_accuracy = evaluate(self.model, test_loader,self.criterion, self.device, get_f1=True)

            print("Train Accuracy",train_accuracy)
            print("Test Accuracy",test_accuracy)

    def get_obs(self):
        if self.current_batch_index >= self.batch_size:
            self.current_batch_index = 0
            self.current_batch_aux_labels = []
            try:
                self.current_batch, self.current_labels = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.train_loader)
                self.current_batch, self.current_labels = next(self.data_iter)

        image =  self.current_batch.cpu()[self.current_batch_index]
        label = self.current_labels.cpu()[self.current_batch_index]

        self.current_batch_index += 1

        return {"image": image, "array": label}

    def reset(self):

        # reset counters
        self.return_ = 0
        self.count = 0

        # Initialize a new main task model from scratch at the start of each episode
        # restore state to cannonical model and optimzizer
        self.model=copy.deepcopy(self.cannonical_model).to(self.device)
        self.optimizer = self.optimizer_func(self.model.parameters())
        if self.optimizer_reload_state is not None:
            self.optimizer.load_state_dict(self.optimizer_reload_state)
        self.scheduler = self.scheduler_func(self.optimizer)
        if self.scheduler_reload_state is not None:
            self.scheduler.load_state_dict(self.scheduler_reload_state)
        # Reset data loader to iterate over batches
        self.reset_data_loader(self.seed)
        self.data_iter = iter(self.train_loader)

        # Return the first batch as observations
        self.current_batch, self.current_labels = next(self.data_iter)
        self.current_batch_aux_labels = []
        return self.get_obs()


    def step_dep(self, action):
        self.count+=1

        # The action is the auxiliary task output generated by the aux_model for the current batch
        inputs, labels = self.current_batch.to(self.device),self.current_labels.to(self.device)

        # Forward pass for the main task model
        self.optimizer.zero_grad()
        class_output, aux_output = self.model(inputs)

        mask=create_mask_from_labels(labels).to(self.device)
        aux_target=mask_softmax(torch.tensor(action).to(self.device),mask,dim=-1)

        loss_class   =  torch.mean(self.model.model_fit(softmax(class_output), labels, pri=True,num_output=20))
        loss_aux  =  torch.mean(self.model.model_fit(softmax(aux_output), aux_target,pri=False, num_output=100))


        info = {"loss_main" : loss_class.item(), "loss_aux": loss_aux.item() }
        # Combine losses (adjust aux_weight as needed)
        loss = loss_class + self.aux_weight * loss_aux
        loss.backward()
        self.optimizer.step()

        # Update observation to next batch
        try:
            self.current_batch, self.current_labels = next(self.data_iter)
            done = False
        except StopIteration:
            done = True
            if self.verbose:
                print("EPISODE FINISHED, steps: ",self.count)

        # Reward is based on reduction in loss, but classification accuracy also may make sense
        with torch.no_grad():
            reward=0
            if self.reward_mode:
                reward_type = "loss"
                # get random sample of dataset with replacement
                current_batch, current_labels = next(self.reward_sampler)
                inputs, labels = current_batch.to(self.device),current_labels.to(self.device)

                # get loss on updated model
                class_output, aux_output = self.model(inputs)
                loss_class_new = self.criterion(class_output, labels)
                if reward_type == "loss":
                    reward =  - loss_class_new.item()
                else:
                    reward =f1_score_from_logits(class_output,labels)

                entropy=0.2*torch.mean(self.model.model_entropy(aux_target))
                reward -= entropy
                self.return_+=reward

        # make sure batch size is correct. May be less on final iteration
        inputs, labels = self.current_batch, self.current_labels
        while inputs.shape[0] < self.batch_size:
            inputs=torch.cat([inputs,inputs],dim=0)
            labels=torch.cat([labels,labels],dim=0)
        inputs=inputs[:self.batch_size]
        labels=labels[:self.batch_size]
        self.current_batch,self.current_labels  = inputs, labels

        return self.get_obs(), reward, done, info

    def step(self, action):
        reward = 0
        done = False
        info = {}

        return self.get_obs(), reward, done, info

    def render(self, mode='human'):
        pass  # Not needed for now