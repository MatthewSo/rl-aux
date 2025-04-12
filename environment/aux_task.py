import gymnasium as gym
import torch
import numpy as np
from gymnasium import spaces
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
import copy
import random
import sys
from utils.evaluate import evaluate
from utils.masked_softmax import create_mask_from_labels, mask_softmax
from utils.randomization import SeededSubsetRandomSampler
from utils.vars import softmax


class AuxTaskEnv(gym.Env):
    def __init__(self, train_dataset, device,model,criterion, optimizer_func, scheduler_func,batch_size=64,pri_dim=20,aux_dim=100,verbose=False, aux_weight = 1):
        super(AuxTaskEnv, self).__init__()
        self.primary_dim=pri_dim
        self.aux_dim=aux_dim
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

        self.observation_space = spaces.Dict({
            "image": image_obs,
        })

        print(self.observation_space)
        self.action_space = spaces.Box(low=-1, high=1, shape=(aux_dim,), dtype=np.float32)

        # Model for the main classification task (reset this each episode)
        self.model=copy.deepcopy(self.cannonical_model).to(self.device)

        # step counter and return counter
        self.count=0
        self.num_batches = 0
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

    def evaluate(self, test_loader):
        # Evaluate the network on train and test sets
        if self.model is not None:
            train_accuracy = evaluate(self.model, self.train_loader,self.criterion, self.device)
            test_accuracy = evaluate(self.model, test_loader,self.criterion, self.device, get_f1=True)

            print("Train Accuracy",train_accuracy)
            print("Test Accuracy",test_accuracy)

    def get_obs(self):
        done = False
        if len(self.current_batch) < self.batch_size:
            image = self.current_batch.cpu()[self.current_batch_index]
            return {"image": image}, True
        if self.current_batch_index >= self.batch_size:
            self.current_batch_index = 0
            self.current_batch_aux_labels = []
            try:
                self.current_batch, self.current_labels = next(self.data_iter)
                done = False
            except StopIteration:
                self.data_iter = iter(self.train_loader)
                self.current_batch, self.current_labels = next(self.data_iter)
                done = True
                print("EPISODE FINISHED, steps: ",self.count)

        image =  self.current_batch.cpu()[self.current_batch_index]

        image = image.numpy().astype(np.float32)

        self.current_batch_index += 1

        return {"image": image}, done

    def reset(self, seed=None):
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

        obs, done = self.get_obs()

        return obs, {}


    def step(self, action, give_reward=True):
        reward = 0
        info = {}
        self.count += 1

        self.current_batch_aux_labels.append(action)

        if len(self.current_batch_aux_labels) >= self.batch_size:
            self.current_batch_aux_labels = [
                torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
                for arr in self.current_batch_aux_labels
            ]
            batch_action = torch.stack(self.current_batch_aux_labels, dim=0)
            self.current_batch_aux_labels = []

            inputs, labels = self.current_batch.to(self.device),self.current_labels.to(self.device)

            self.optimizer.zero_grad()
            primary_output, aux_output = self.model(inputs)

            mask=create_mask_from_labels(labels, num_classes=self.primary_dim, num_features=self.aux_dim ).to(self.device)
            aux_target=mask_softmax(torch.tensor(batch_action).to(self.device),mask,dim=-1)

            loss_class = torch.mean(self.model.model_fit(softmax(primary_output), labels, pri=True,num_output=self.primary_dim, device=self.device))
            loss_aux = torch.mean(self.model.model_fit(softmax(aux_output), aux_target,pri=False, num_output=self.aux_dim, device=self.device))

            info = {"loss_main" : loss_class.item(), "loss_aux": loss_aux.item() }
            if self.verbose:
                print("loss_main",loss_class.item())

            #loss = loss_class + self.aux_weight * loss_aux
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
                        print("loss",loss_class_new.item())
                    reward =  - loss_class_new.item()
                    entropy=0.2*torch.mean(self.model.model_entropy(aux_target))
                    reward -= entropy
                    self.return_ +=reward
                    self.num_batches += 1

        obs, done = self.get_obs()
        return obs, reward, done, False, info

    def render(self, mode='human'):
        pass  # Not needed for now

    def train_label_network_with_rl(self, model):
        episode_length = len(self.train_loader) * self.batch_size # Total steps in an episode
        model.learn(total_timesteps=episode_length)

    def train_main_network(self, model):
        self.reward_mode=False
        obs, _info = self.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, _info = self.step(action)
        self.update()
        self.reward_mode=True