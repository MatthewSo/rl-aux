import os
import torch.nn.functional as F

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

WEIGHT_DIMS = 21

class AuxTaskEnv(gym.Env):
    def __init__(self, train_dataset, device,model,criterion, optimizer_func, scheduler_func, batch_size, image_shape=(3,32,32),pri_dim=20,aux_dim=100,verbose=False, aux_weight = 1, learn_weights=False, save_path='./' ):
        super(AuxTaskEnv, self).__init__()
        self.primary_dim=pri_dim
        self.aux_dim=aux_dim
        self.batch_size=batch_size
        self.train_dataset=train_dataset
        self.cannonical_model=copy.deepcopy(model)
        self.criterion = criterion
        self.verbose=verbose
        self.aux_weight=aux_weight
        self.learn_weights = learn_weights
        self.save_path=save_path

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
        self.current_batch_weights = []

        # create endless sampler for reward sampling
        sampler = RandomSampler(train_dataset, replacement=True,num_samples=sys.maxsize )
        self.reward_sampler = iter( DataLoader(train_dataset, batch_size=256, sampler=sampler))

        # Define action and observation space
        image_obs = spaces.Box(low=0, high=1, shape=image_shape, dtype=np.float32)

        self.observation_space = spaces.Dict({
            "image": image_obs,
        })

        log_print(self.observation_space)
        self.action_space = spaces.MultiDiscrete(np.array([aux_dim, WEIGHT_DIMS], dtype=np.int64))

        # Model for the main classification task (reset this each episode)
        self.model=copy.deepcopy(self.cannonical_model).to(self.device)

        # step counter and return counter
        self.count=0
        self.num_batches = 0
        self.return_ = 0

    def weight_idx_to_float(self, idx, num_weight_bins):
        step = 1.0 / (num_weight_bins - 1)
        return idx * step

    def randomize_seed(self):
        self.seed = random.randint(0, 2**32 - 1)

    def reset_data_loader(self,seed):
        # Define the indices for the dataset
        indices = list(range(len(self.train_dataset)))
        sampler = SeededSubsetRandomSampler(indices, seed=seed)

        self.train_loader  = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler)

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

        # Save the cannonical model
        torch.save(self.cannonical_model.state_dict(), save_path + "/cannonical_model.pth")

        # Save the agent stable baselines 3
        agent.save(save_path + "/agent")

    def \
            update(self):
        log_print(self.scheduler.state_dict())
        log_print(self.optimizer.state_dict())
        self.scheduler.step()
        self.cannonical_model=copy.deepcopy(self.model)
        self.optimizer_reload_state=copy.deepcopy( self.optimizer.state_dict())
        self.scheduler_reload_state=copy.deepcopy( self.scheduler.state_dict())
        self.randomize_seed()


    def evaluate(self, test_loader):
        # Evaluate the network on train and test sets
        if self.model is not None:
            train_accuracy = evaluate(self.model, self.train_loader,self.criterion, self.device)
            test_accuracy = evaluate(self.model, test_loader,self.criterion, self.device, get_f1=True)

            log_print("Train Accuracy",train_accuracy)
            log_print("Test Accuracy",test_accuracy)

    def get_obs(self):
        done = False
        if len(self.current_batch) < self.batch_size:
            image = self.current_batch.cpu()[self.current_batch_index]
            return {"image": image}, True
        if self.current_batch_index >= self.batch_size:
            self.current_batch_index = 0
            self.current_batch_aux_labels = []
            self.current_batch_weights = []
            try:
                self.current_batch, self.current_labels = next(self.data_iter)
                done = False
            except StopIteration:
                self.data_iter = iter(self.train_loader)
                self.current_batch, self.current_labels = next(self.data_iter)
                done = True
                log_print("EPISODE FINISHED, steps: ",self.count)

        image =  self.current_batch.cpu()[self.current_batch_index]

        image = image.numpy().astype(np.float32)

        self.current_batch_index += 1

        return {"image": image}, done

    def reset(self, seed=None, options=None):
        if self.verbose:
            log_print("Resetting environment")

        # reset counters
        self.return_ = 0
        self.count = 0
        self.num_batches = 0

        # Initialize a new main task model from scratch at the start of each episode
        # restore state to cannonical model and optimzizer
        self.model=copy.deepcopy(self.cannonical_model).to(self.device)
        self.optimizer = self.optimizer_func(self.model)
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
        self.current_batch_weights = []

        obs, done = self.get_obs()

        return obs, {}


    def step(self, action, give_reward=True):
        reward = 0
        info = {}
        self.count += 1

        label, weight = action

        self.current_batch_aux_labels.append(torch.as_tensor(label, dtype=torch.long))
        self.current_batch_weights.append(torch.as_tensor(weight, dtype=torch.long))

        if len(self.current_batch_aux_labels) >= self.batch_size:
            self.num_batches += 1

            self.current_batch_aux_labels = [
                torch.from_numpy(x) if isinstance(x, np.ndarray) else x
                for x in self.current_batch_aux_labels
            ]
            self.current_batch_weights = [
                torch.from_numpy(x) if isinstance(x, np.ndarray) else x
                for x in self.current_batch_weights
            ]

            aux_labels = torch.stack(self.current_batch_aux_labels, dim=0)
            weights    = torch.stack(self.current_batch_weights, dim=0)

            weights = torch.tensor(
                self.weight_idx_to_float(weights.cpu().numpy(), WEIGHT_DIMS),
                dtype=torch.float32,
                device=self.device,
            )

            self.current_batch_aux_labels = []
            self.current_batch_weights = []

            inputs, labels = self.current_batch.to(self.device),self.current_labels.to(self.device)

            self.optimizer.zero_grad()
            primary_output, aux_output = self.model(inputs)

            mask=create_mask_from_labels(labels, num_classes=self.primary_dim, num_features=self.aux_dim ).to(self.device)
            B, C = mask.shape                       # C = 1825
            device = mask.device

            # 1. one-hot encode → (B, C)
            x = F.one_hot(aux_labels, num_classes=C).float().to(device)

            # 2. masked soft-max over the class dimension
            aux_target = mask_softmax(x, mask, dim=1)
            # aux_labels = aux_labels.to(self.device).unsqueeze(1).float()
            # aux_target=mask_softmax(torch.tensor(aux_labels).to(self.device),mask,dim=1)
            #aux_target=mask_softmax(torch.tensor(aux_labels).to(self.device),mask,dim=-1)

            loss_class = torch.mean(self.model.model_fit(primary_output, labels, pri=True,num_output=self.primary_dim, device=self.device))
            loss_aux_individual = self.model.model_fit(aux_output, aux_target,pri=False, num_output=self.aux_dim, device=self.device)

            if self.learn_weights:
                weight_factors = torch.pow(2.0, 8.0 * weights - 3.0)
                if self.verbose:
                    if self.num_batches % 100 == 0:
                        log_print("weight_factors",weight_factors)

                loss_aux = torch.mean(loss_aux_individual * weight_factors)
                loss = loss_class + loss_aux
            else:
                loss_aux = torch.mean(loss_aux_individual)
                loss = loss_class + self.aux_weight * loss_aux

            info = {"loss_main" : loss_class.item(), "loss_aux": loss_aux.item() }
            if self.verbose:
                if self.num_batches % 50 == 0:
                    log_print("num_batches",self.num_batches)
                    log_print("loss",loss_class.item())
                    log_print("loss_aux",loss_aux.item())

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
                    entropy=0.2*torch.mean(self.model.model_entropy(aux_target))
                    reward -= entropy
                    self.return_ +=reward

                #self.reset()

        obs, done = self.get_obs()
        return obs, reward, done, False, info

    def render(self, mode='human'):
        pass  # Not needed for now

    def train_rl_network_with_rl(self, model, ratio=1):
        episode_length = len(self.train_loader) * self.batch_size # Total steps in an episode
        episode_length = int(episode_length * ratio)
        model.learn(total_timesteps=episode_length)

    def train_main_network(self, model):
        self.reward_mode=False
        obs, _info = self.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, _info = self.step(action, give_reward=False)
        self.update()
        self.reward_mode=True