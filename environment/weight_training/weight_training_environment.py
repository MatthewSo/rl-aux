import os
import gymnasium as gym
import torch
import numpy as np
from gymnasium import spaces
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
import copy
import random
import sys

from utils.log import log_print
from utils.masked_softmax import create_mask_from_labels, mask_softmax
from utils.randomization import SeededSubsetRandomSampler
import torch.nn.functional as F

class WeightTuningEnv(gym.Env):
    def __init__(self, train_dataset, device, model, labeler, criterion, optimizer_func,
                 scheduler_func, batch_size, image_shape=(3,32,32), pri_dim=20, aux_dim=100,
                 verbose=False, save_path="./"):
        super(WeightTuningEnv, self).__init__()
        self.primary_dim = pri_dim
        self.aux_dim = aux_dim
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.cannonical_model = copy.deepcopy(model)
        self.labeler = labeler
        self.criterion = criterion
        self.verbose = verbose
        self.save_path = save_path
        self.optimizer_func = optimizer_func
        self.scheduler_func = scheduler_func
        self.optimizer_reload_state = None
        self.optimizer = None
        self.scheduler_reload_state = None
        self.scheduler = None
        self.device = device
        self.randomize_seed()
        self.reset_data_loader(self.seed)
        self.current_batch = None
        self.current_batch_index = 0
        self.current_batch_weights = []
        self.current_batch_aux_labels = []
        sampler = RandomSampler(train_dataset, replacement=True, num_samples=sys.maxsize)
        self.reward_sampler = iter(DataLoader(train_dataset, batch_size=256, sampler=sampler))
        image_obs = spaces.Box(low=0, high=1, shape=image_shape, dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image": image_obs,
        })
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.model = copy.deepcopy(self.cannonical_model).to(self.device)
        self.count = 0
        self.num_batches = 0
        self.return_ = 0
    def randomize_seed(self):
        self.seed = random.randint(0, 2**32 - 1)
    def reset_data_loader(self, seed):
        indices = list(range(len(self.train_dataset)))
        sampler = SeededSubsetRandomSampler(indices, seed=seed)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler)
    def get_obs(self):
        done = False
        if len(self.current_batch) < self.batch_size:
            image = self.current_batch.cpu()[self.current_batch_index]
            action, state = self.labeler.predict({"image": image})
            aux_task_action, weight_task_action = action
            self.current_batch_aux_labels.append(torch.as_tensor(aux_task_action, dtype=torch.long))
            return {"image": image}, True
        if self.current_batch_index >= self.batch_size:
            self.current_batch_index = 0
            self.current_batch_weights = []
            try:
                self.current_batch, self.current_labels = next(self.data_iter)
                done = False
            except StopIteration:
                self.data_iter = iter(self.train_loader)
                self.current_batch, self.current_labels = next(self.data_iter)
                done = True
        image = self.current_batch.cpu()[self.current_batch_index]
        image = image.numpy().astype(np.float32)

        action, state = self.labeler.predict({"image": image})
        aux_task_action, weight_task_action = action
        self.current_batch_aux_labels.append(torch.as_tensor(aux_task_action, dtype=torch.long))

        self.current_batch_index += 1
        return {"image": image}, done
    def reset(self, seed=None):
        self.return_ = 0
        self.count = 0
        self.num_batches = 0
        self.model = copy.deepcopy(self.cannonical_model).to(self.device)
        self.optimizer = self.optimizer_func(self.model)
        if self.optimizer_reload_state is not None:
            self.optimizer.load_state_dict(self.optimizer_reload_state)
        self.scheduler = self.scheduler_func(self.optimizer)
        if self.scheduler_reload_state is not None:
            self.scheduler.load_state_dict(self.scheduler_reload_state)
        self.reset_data_loader(self.seed)
        self.data_iter = iter(self.train_loader)
        self.current_batch, self.current_labels = next(self.data_iter)
        self.current_batch_weights = []
        self.current_batch_aux_labels = []
        obs, done = self.get_obs()
        return obs, {}
    def step(self, action, give_reward=True):
        reward = 0
        info = {}
        self.count += 1
        #weight = float(np.clip(action, 0.0, 1.0))
        self.current_batch_weights.append(torch.tensor(weight, dtype=torch.long))
        if len(self.current_batch_weights) >= self.batch_size:
            self.num_batches += 1

            inputs, labels = self.current_batch.to(self.device),self.current_labels.to(self.device)
            self.current_batch_aux_labels = [
                torch.from_numpy(x) if isinstance(x, np.ndarray) else x
                for x in self.current_batch_aux_labels
            ]
            self.current_batch_weights = [
                torch.from_numpy(x) if isinstance(x, np.ndarray) else x
                for x in self.current_batch_weights
            ]

            mask = create_mask_from_labels(labels, num_classes=self.primary_dim, num_features=self.aux_dim).to(self.device)
            weights = torch.stack(self.current_batch_weights, dim=0).to(self.device)
            aux_labels = torch.stack(self.current_batch_aux_labels, dim=0).to(self.device)
            aux_labels_onehot = F.one_hot(aux_labels.long(), num_classes=self.aux_dim).float()

            aux_target = mask_softmax(aux_labels_onehot, mask, dim=-1)
            self.current_batch_aux_labels = []
            self.current_batch_weights = []

            self.optimizer.zero_grad()
            primary_output, aux_output = self.model(inputs)
            loss_class = self.criterion(primary_output, labels)

            loss_aux_individual = self.model.model_fit(aux_output, aux_target, device=self.device, pri=False, num_output=self.aux_dim)
            weight_factors = torch.pow(2.0, 10.0 * weights - 5.0)

            loss_aux = torch.mean(loss_aux_individual * weight_factors)

            if self.verbose:
                if self.num_batches % 50 == 0:
                    log_print("num_batches",self.num_batches)
                    log_print("loss",loss_class.item())
                    log_print("loss_aux",loss_aux.item())
                if self.num_batches % 100 == 0:
                    # print first 30 weight factors
                    log_print("weight_factors", weight_factors[:30])
                    log_print("weights", weights[:30])

            loss = loss_class + loss_aux
            loss.backward()
            self.optimizer.step()
            if give_reward:
                with torch.no_grad():
                    reward_batch, reward_labels = next(self.reward_sampler)
                    reward_batch, reward_labels = reward_batch.to(self.device), reward_labels.to(self.device)
                    class_output, _ = self.model(reward_batch)
                    loss_class_new = self.criterion(class_output, reward_labels)
                    reward = -loss_class_new.item()
                    self.return_ += reward
        obs, done = self.get_obs()
        return obs, reward, done, False, info
    def render(self, mode='human'):
        pass

    def train_rl_network_with_rl(self, model, ratio=1):
        episode_length = len(self.train_loader) * self.batch_size
        episode_length = int(episode_length * ratio)
        model.learn(total_timesteps=episode_length)

    def update(self):
        self.scheduler.step()
        self.cannonical_model = copy.deepcopy(self.model)
        self.optimizer_reload_state = copy.deepcopy(self.optimizer.state_dict())
        self.scheduler_reload_state = copy.deepcopy(self.scheduler.state_dict())
        self.randomize_seed()

    def train_main_network(self, weight_agent):
        obs, _ = self.reset()
        done = False
        while not done:
            action, _ = weight_agent.predict(obs, deterministic=True)
            obs, _, done, _, _ = self.step(action, give_reward=False)
        self.update()