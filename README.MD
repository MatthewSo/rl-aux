# RL-Aux: Reinforcement Learning for Auxiliary Tasks

This repo implements an RL framework to learn auxiliary tasks using PPO (Proximal Policy Optimization) for priamry classification task performance improvement. 
## Features
- **Custom PPO Agent**: Implements a PPO agent for learning auxiliary tasks.
- **Custom Gymnasium Environment**: Defines a custom environment for auxiliary task learning.
- **Auxiliary-Task Supported Primary Model**: Integrates auxiliary tasks into the primary VGG16 model for improved performance.
- **Standardized Dataloaders**: Provides standardized dataloaders for CIFAR-100, CIFAR-10, PLACES365, and SVHN datasets.
- **MAXL Samples**: Implements MAXL samples for auxiliary task learning.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MatthewSo/rl-aux.git
   cd rl-aux
   ```

2. Create conda environment from environment.yaml:
   ```bash
   conda env create -f environment.yaml
   ```

3. Ensure you have the required datasets downloaded in the `./data` directory. You can use the download flag in the dataset loaders to automatically download the datasets.

## Usage

### Run Trainings
The top level directory provides predefined entry point scripts for training the model with different datasets and configurations.

For example to train the base VGG16 model on 20-Superclass CIFAR-100 with weight-aware auxiliary RL auxiliary tasks, run:
```
conda activate rl-aux

python cifar100_20_entry_point_learn_weights.py
```
There are several at the top level for different configurations and tests. The training scripts generate logs and checkpoints in the run-specific folder. You can also find the configuration of the run in the parameters.json file in the run folder.

The entry points are named with keyworks for the type of training they perform:
- `learn_weights`: Indicates that the auxiliary task weights are learned.
- `equal`: Indicates that the auxiliary task weights are equal. (This is the default RL training mode)
- `no_aux`: Indicates that the auxiliary task weights are not learned and the main network is not trained with an auxiliary task.

NOTE: The Places365 dataset is under active development and the entry point for it may not yet be stable.

### Run MAXL
It is easiest to run MAXL as a module from the top directory. There are other scripts to run different versions of maxl in the legacy folder.
```
conda activate rl-aux

python -m legacy.model_vgg_maxl.py
```


### Parameters
You can modify the following parameters in the entry point scripts:
- `BATCH_SIZE`: Batch size for training.
- `TOTAL_EPOCH`: Number of training epochs.
- `PRIMARY_LEARNING_RATE`: Learning rate for the primary model.
- `PPO_LEARNING_RATE`: Learning rate for the PPO agent.
- `LEARN_WEIGHTS`: Boolean to indicate if the auxiliary task weights should be learned.
- `AUX_WEIGHT`: Weight for auxiliary task loss. (Will be ignored if LEARN_WEIGHTS is set to True)
- `TRAIN_RATIO`: Ratio of primary to auxiliary task training.

### Device Configuration
The device (CPU or GPU) can be set in the entry point scripts:
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## Project Structure
- `datasets/`: Contains dataset loaders and transformations.
- `environment/`: Defines the auxiliary task environment.
- `networks/`: Includes PPO and primary model architectures.
- `train/`: Training scripts for the auxiliary agent.
- `utils/`: Utility functions for logging and path management.
- `legacy/`: Contains MAXL implementations and scripts.
