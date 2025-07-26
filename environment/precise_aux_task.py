import os
from tabnanny import verbose

import gymnasium as gym
import torch
import numpy as np
from gymnasium import spaces
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
import copy
import random
import sys
from utils.evaluate import evaluate
from utils.log import log_print
from utils.masked_softmax import create_mask_from_labels, mask_softmax
from utils.randomization import SeededSubsetRandomSampler
from utils.vars import softmax
import torch.nn.functional as F

class PreciseAuxTaskEnv(gym.Env):
    def __init__(self, train_dataset, test_dataset, device, model, criterion, optimizer_func,
                 scheduler_func, image_shape, pri_dim, hierarchy_factor, rl_analysis_batch_size,
                 verbose=False, aux_weight = 1, save_path='./' ):
        super(PreciseAuxTaskEnv, self).__init__()
        self.batch_size = 1

        self.primary_dim=pri_dim
        self.hierarchy_factor=hierarchy_factor

        self.train_dataset=train_dataset
        self.test_dataset=test_dataset

        self.criterion = criterion
        self.optimizer_func=optimizer_func
        self.scheduler_func=scheduler_func

        self.verbose=verbose
        self.aux_weight=aux_weight
        self.save_path=save_path

        # optimizer and its reload state
        self.optimizer_reload_state=None
        self.optimizer = None
        self.scheduler_reload_state = None
        self.scheduler = None

        # Initialize environment variables
        self.device = device

        # initialize dataloader with seed
        self.randomize_seed()
        self.reset_data_loaders(self.seed)
        self.current_image = None

        # create endless sampler for reward sampling
        sampler = RandomSampler(train_dataset, replacement=True, num_samples=sys.maxsize)
        self.reward_sampler = iter(DataLoader(train_dataset, batch_size=rl_analysis_batch_size, sampler=sampler))

        # Define action and observation space
        image_obs = spaces.Box(low=0, high=1, shape=image_shape, dtype=np.float32)

        self.observation_space = spaces.Dict({
            "image": image_obs,
        })

        self.action_space = spaces.Discrete(self.hierarchy_factor)

        # Model for the main classification task (reset this each episode)
        self.canonical_model=copy.deepcopy(model)
        self.model=copy.deepcopy(self.canonical_model).to(self.device)

        # step counter and return counter
        self.count = 0
        self.num_batches = 0
        self.return_ = 0

    def randomize_seed(self):
        self.seed = random.randint(0, 2**32 - 1)

    def reset_data_loaders(self,seed):
        # Define the indices for the dataset
        train_indices = list(range(len(self.train_dataset)))
        train_sampler = SeededSubsetRandomSampler(train_indices, seed=seed)
        self.train_loader  = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler)

        test_indices = list(range(len(self.test_dataset)))
        test_sampler = SeededSubsetRandomSampler(test_indices, seed=seed)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=test_sampler)

        self.train_data_iter = iter(self.train_loader)
        self.test_data_iter = iter(self.test_loader)

    def get_stats(self):
        log_print("return:",self.return_)
        log_print("len",self.count)

    def save(self, agent, save_path=None):
        if save_path is None:
            save_path = self.save_path

        # make path if needed
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save the model and optimizer state
        torch.save(self.model.state_dict(), save_path + "/model.pth")
        torch.save(self.optimizer.state_dict(), save_path + "/optimizer.pth")
        torch.save(self.scheduler.state_dict(), save_path + "/scheduler.pth")

        # Save the canonical model
        torch.save(self.canonical_model.state_dict(), save_path + "/canonical_model.pth")

        # Save the agent stable baselines 3
        agent.save(save_path + "/agent")

    def update(self):
        log_print(self.scheduler.state_dict())
        log_print(self.optimizer.state_dict())
        self.scheduler.step()
        self.canonical_model=copy.deepcopy(self.model)
        self.optimizer_reload_state=copy.deepcopy( self.optimizer.state_dict())
        self.scheduler_reload_state=copy.deepcopy( self.scheduler.state_dict())
        self.randomize_seed()


    def evaluate(self):
        # Evaluate the network on train and test sets
        if self.model is not None:
            train_accuracy = evaluate(self.model, self.train_loader,self.criterion, self.device)
            test_accuracy = evaluate(self.model, self.test_loader,self.criterion, self.device, get_f1=True)

            log_print("Train Accuracy",train_accuracy)
            log_print("Test Accuracy",test_accuracy)

    def get_obs(self):
        done = False
        try:
            self.current_image, self.current_label = next(self.train_data_iter)
        except StopIteration:
            done = True
            self.train_data_iter = iter(self.train_loader)
            self.current_image, self.current_label = next(self.train_data_iter)

        obs_img = self.current_image[0].detach().cpu().numpy().astype(np.float32)
        return {"image": obs_img}, done

    def reset(self, seed=None):
        if self.verbose:
            log_print("Resetting environment")

        # reset counters
        self.return_ = 0
        self.count = 0
        self.num_batches = 0

        # Initialize a new main task model from scratch at the start of each episode
        # restore state to canonical model and optimzizer
        self.model=copy.deepcopy(self.canonical_model).to(self.device)
        self.optimizer = self.optimizer_func(self.model)
        if self.optimizer_reload_state is not None:
            self.optimizer.load_state_dict(self.optimizer_reload_state)
        self.scheduler = self.scheduler_func(self.optimizer)
        if self.scheduler_reload_state is not None:
            self.scheduler.load_state_dict(self.scheduler_reload_state)
        # Reset data loader to iterate over batches
        self.reset_data_loaders(self.seed)

        self.current_image, self.current_label = None, None

        obs, done = self.get_obs()

        return obs, {}


    def step(self, action, give_reward=True, revert_model=True):
        reward = 0
        info = {}
        self.count += 1

        if revert_model:
            # Revert the model to the canonical model
            self.model = copy.deepcopy(self.canonical_model).to(self.device)
            self.optimizer = self.optimizer_func(self.model)
            if self.optimizer_reload_state is not None:
                self.optimizer.load_state_dict(self.optimizer_reload_state)
            self.scheduler = self.scheduler_func(self.optimizer)
            if self.scheduler_reload_state is not None:
                self.scheduler.load_state_dict(self.scheduler_reload_state)

        input = self.current_image.to(self.device, non_blocking=True)
        label = self.current_label.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        primary_output, aux_output = self.model(input)

        log_print("Input shape:", input.shape)
        log_print("Label shape:", label.shape)
        log_print("Primary output shape:", primary_output.shape)
        log_print("Auxiliary output shape:", aux_output.shape)
        log_print("Action taken:", action)

        action_idx = torch.as_tensor(action, dtype=torch.long, device=self.device).unsqueeze(0)
        log_print("Action index shape:", action_idx.shape)
        flat_idx = label.to(torch.long) * self.hierarchy_factor + action_idx
        log_print("Flat index shape:", flat_idx.shape)
        aux_target = torch.zeros((aux_output.size(0), self.hierarchy_factor * self.primary_dim), device=self.device)
        log_print("Auxiliary target shape:", aux_target.shape)
        aux_target.scatter_(1, flat_idx.unsqueeze(1), 1.0)

        # ADDBACK: We can add back in the mask softmax if needed
        #mask=create_mask_from_labels(labels, num_classes=self.primary_dim, num_features=self.aux_dim ).to(self.device)
        #aux_target=mask_softmax(torch.tensor(aux_target).to(self.device),mask,dim=-1)

        loss_class = torch.mean(self.model.model_fit(primary_output, label, pri=True,num_output=self.primary_dim, device=self.device))
        loss_aux = torch.mean(self.model.model_fit(aux_output, aux_target,pri=False, num_output=self.hierarchy_factor * self.primary_dim, device=self.device))

        info = {"loss_main" : loss_class.item(), "loss_aux": loss_aux.item() }
        if self.verbose:
            if self.num_batches % 50 == 0:
                log_print("num_batches",self.num_batches)
                log_print("loss",loss_class.item())
                log_print("loss_aux",loss_aux.item())

        loss = loss_class + self.aux_weight * loss_aux
        if self.aux_weight == 0:
            loss = loss_class
        loss.backward()
        self.optimizer.step()

        if give_reward:
            with torch.no_grad():
                reward_batch, reward_labels = next(self.reward_sampler)
                reward_batch, reward_labels = reward_batch.to(self.device),reward_labels.to(self.device)

                class_output, aux_output = self.model(reward_batch)
                loss_class_new = self.criterion(class_output, reward_labels)
                if self.verbose:
                    if self.num_batches % 50 == 0:
                        log_print("num_batches",self.num_batches)
                        log_print("loss",loss_class_new.item())
                reward =  - loss_class_new.item()

                # ADDBACK: We can add back in the entropy if needed
                # entropy=0.2*torch.mean(self.model.model_entropy(aux_target))
                # reward -= entropy
                # self.return_ +=reward

        obs, done = self.get_obs()
        return obs, reward, done, False, info

    def render(self, mode='human'):
        pass  # Not needed for now

    def train_rl_network_with_rl(self, model, ratio=1):
        episode_length = len(self.train_loader) * self.batch_size
        episode_length = int(episode_length * ratio)
        model.learn(total_timesteps=episode_length)

    def train_main_network(self, model):
        self.reward_mode=False
        obs, _info = self.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, _info = self.step(action, give_reward=False, revert_model=False)
        self.update()
        self.reward_mode=True