import pickle
from collections import OrderedDict

from dataset_loaders.cifar10 import CIFAR10

print("importing collections")
from dataset_loaders.transforms import cifar_trans_test, cifar_trans_train

print("importing transforms")
import numpy as np
print("importing numpy")

import torch
print("importing torch")
import torch.nn as nn
print("importing nn")
import torchvision.transforms as transforms
print("importing transforms")
import torch.optim as optim
print("importing optim")
import torch.nn.functional as F
print("importing functional")
import torch.utils.data.sampler as sampler

print("importing sampler")
from train.model.performance import EpochPerformance
print("importing EpochPerformance")
from utils.log import change_log_location, log_print
print("importing log")
from utils.path_name import create_path_name
print("importing path_name")

AUX_WEIGHT = 0

save_path = create_path_name(
    agent_type="MAXL",
    primary_model_type="VGG",
    train_ratio=1,
    aux_weight=AUX_WEIGHT,
    observation_feature_dimensions=0,
    dataset="CIFAR10",
    learn_weights=True,
)
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

train_set = CIFAR10(
    root="./data/cifar10",
    train=True,
    transform=cifar_trans_train
)
test_set = CIFAR10(
    root="./data/cifar10",
    train=False,
    transform=cifar_trans_test
)

PRIMARY_CLASS = 10
AUXILIARY_CLASS = 50

change_log_location(save_path)
epoch_performances=[]

class LabelGenerator(nn.Module):
    def __init__(self, psi):
        super(LabelGenerator, self).__init__()
        """
            label-generation network:
            takes the input and generates auxiliary labels with masked softmax for an auxiliary task.
        """
        filter = [64, 128, 256, 512, 512]
        self.class_nb = psi

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
        index = index.to(device)
        for i in range(len(self.class_nb)):
            index[i, int(np.sum(self.class_nb[:i])):np.sum(self.class_nb[:i+1])] = 1
        mask = index[y].to(device)

        predict = self.classifier(g_block5.view(g_block5.size(0), -1))
        label_pred = self.mask_softmax(predict, mask, dim=1)

        weight = self.weight_head(g_block5.view(g_block5.size(0), -1))

        return label_pred, weight


class VGG16(nn.Module):
    def __init__(self, psi):
        super(VGG16, self).__init__()
        """
            multi-task network:
            takes the input and predicts primary and auxiliary labels (same network structure as in human)
        """
        filter = [64, 128, 256, 512, 512]

        # define convolution block in VGG-16
        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        # primary task prediction
        self.classifier1 = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], len(psi)),
            nn.Softmax(dim=1)
        )

        # auxiliary task prediction
        self.classifier2 = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], int(np.sum(psi))),
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

    # define forward conv-layer (will be used in second-derivative step)
    def conv_layer_ff(self, input, weights, index):
        if index < 3:
            net = F.conv2d(input, weights['block{:d}.0.weight'.format(index)], weights['block{:d}.0.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['block{:d}.1.weight'.format(index)], weights['block{:d}.1.bias'.format(index)],
                               training=True)
            net = F.relu(net, inplace=True)
            net = F.conv2d(net, weights['block{:d}.3.weight'.format(index)], weights['block{:d}.3.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['block{:d}.4.weight'.format(index)], weights['block{:d}.4.bias'.format(index)],
                               training=True)
            net = F.relu(net, inplace=True)
            net = F.max_pool2d(net, kernel_size=2, stride=2, )
        else:
            net = F.conv2d(input, weights['block{:d}.0.weight'.format(index)], weights['block{:d}.0.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['block{:d}.1.weight'.format(index)], weights['block{:d}.1.bias'.format(index)],
                               training=True)
            net = F.relu(net, inplace=True)
            net = F.conv2d(net, weights['block{:d}.3.weight'.format(index)], weights['block{:d}.3.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['block{:d}.4.weight'.format(index)], weights['block{:d}.4.bias'.format(index)],
                               training=True)
            net = F.relu(net, inplace=True)
            net = F.conv2d(net, weights['block{:d}.6.weight'.format(index)], weights['block{:d}.6.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['block{:d}.7.weight'.format(index)], weights['block{:d}.7.bias'.format(index)],
                               training=True)
            net = F.relu(net, inplace=True)
            net = F.max_pool2d(net, kernel_size=2, stride=2)
        return net

    # define forward fc-layer (will be used in second-derivative step)
    def dense_layer_ff(self, input, weights, index):
        net = F.linear(input, weights['classifier{:d}.0.weight'.format(index)], weights['classifier{:d}.0.bias'.format(index)])
        net = F.relu(net, inplace=True)
        net = F.linear(net, weights['classifier{:d}.2.weight'.format(index)], weights['classifier{:d}.2.bias'.format(index)])
        net = F.softmax(net, dim=1)
        return net

    def forward(self, x, weights=None):
        """
            if no weights given, use the direct training strategy and update network paramters
            else retain the computational graph which will be used in second-derivative step
        """
        if weights is None:
            g_block1 = self.block1(x)
            g_block2 = self.block2(g_block1)
            g_block3 = self.block3(g_block2)
            g_block4 = self.block4(g_block3)
            g_block5 = self.block5(g_block4)

            t1_pred = self.classifier1(g_block5.view(g_block5.size(0), -1))
            t2_pred = self.classifier2(g_block5.view(g_block5.size(0), -1))

        else:
            g_block1 = self.conv_layer_ff(x, weights, 1)
            g_block2 = self.conv_layer_ff(g_block1, weights, 2)
            g_block3 = self.conv_layer_ff(g_block2, weights, 3)
            g_block4 = self.conv_layer_ff(g_block3, weights, 4)
            g_block5 = self.conv_layer_ff(g_block4, weights, 5)

            t1_pred = self.dense_layer_ff(g_block5.view(g_block5.size(0), -1), weights, 1)
            t2_pred = self.dense_layer_ff(g_block5.view(g_block5.size(0), -1), weights, 2)

        return t1_pred, t2_pred

    def model_fit(self, x_pred, x_output, pri=True, num_output=3):
        if not pri:
            # generated auxiliary label is a soft-assignment vector (no need to change into one-hot vector)
            x_output_onehot = x_output
        else:
            # convert a single label into a one-hot vector
            x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
            x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

        # apply focal loss
        loss = x_output_onehot * (1 - x_pred)**2 * torch.log(x_pred + 1e-20)
        return torch.sum(-loss, dim=1)

    def model_entropy(self, x_pred1):
        # compute entropy loss
        x_pred1 = torch.mean(x_pred1, dim=0)
        loss1 = x_pred1 * torch.log(x_pred1 + 1e-20)
        return torch.sum(loss1)


# load CIFAR100 dataset
trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),

])
trans_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),

])

batch_size = 100
dataloader_train = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=True
)

kwargs = {'num_workers': 1, 'pin_memory': True}

# define label-generation model,
# and optimiser with learning rate 1e-3, drop half for every 50 epochs, weight_decay=5e-4,
psi = [5]*PRIMARY_CLASS  # for each primary class split into 5 auxiliary classes, with total 100 auxiliary classes
LabelGenerator = LabelGenerator(psi=psi).to(device)
gen_optimizer = optim.SGD(LabelGenerator.parameters(), lr=1e-3, weight_decay=5e-4)
gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=50, gamma=0.5)

# define parameters
total_epoch = 200
train_batch = len(dataloader_train)
test_batch = len(dataloader_test)

# define multi-task network, and optimiser with learning rate 0.01, drop half for every 50 epochs
VGG16_model = VGG16(psi=psi).to(device)
optimizer = optim.SGD(VGG16_model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
avg_cost = np.zeros([total_epoch, 9], dtype=np.float32)
vgg_lr = 0.01  # define learning rate for second-derivative step (theta_1^+)
k = 0
for index in range(total_epoch):
    cost = np.zeros(4, dtype=np.float32)

    # drop the learning rate with the same strategy in the multi-task network
    # note: not necessary to be consistent with the multi-task network's parameter,
    # it can also be learned directly from the network
    if (index + 1) % 50 == 0:
        vgg_lr = vgg_lr * 0.5

    # evaluate training data (training-step, update on theta_1)
    VGG16_model.train()
    cifar100_train_dataset = iter(dataloader_train)
    for i in range(train_batch):
        train_data, train_label = next(cifar100_train_dataset)
        train_label = train_label.type(torch.LongTensor)
        train_data, train_label = train_data.to(device), train_label.to(device)
        train_pred1, train_pred2 = VGG16_model(train_data)
        train_pred3, aux_weight = LabelGenerator(train_data, train_label)  # generate auxiliary labels

        # reset optimizers with zero gradient
        optimizer.zero_grad()
        gen_optimizer.zero_grad()

        # choose level 2/3 hierarchy, 20-class (gt) / 100-class classification (generated by labelgeneartor)
        train_loss1 = VGG16_model.model_fit(train_pred1, train_label, pri=True, num_output=PRIMARY_CLASS)
        train_loss2 = VGG16_model.model_fit(train_pred2, train_pred3, pri=False, num_output=AUXILIARY_CLASS)

        # compute cosine similarity between gradients from primary and auxiliary loss
        grads1 = torch.autograd.grad(torch.mean(train_loss1), VGG16_model.parameters(), retain_graph=True, allow_unused=True)
        grads2 = torch.autograd.grad(torch.mean(train_loss2), VGG16_model.parameters(), retain_graph=True, allow_unused=True)
        cos_mean = 0
        for l in range(len(grads1) - 8):  # only compute on shared representation (ignore task-specific fc-layers)
            grads1_ = grads1[l].view(grads1[l].shape[0], -1)
            grads2_ = grads2[l].view(grads2[l].shape[0], -1)
            cos_mean += torch.mean(F.cosine_similarity(grads1_, grads2_, dim=-1)) / (len(grads1) - 8)
        # cosine similarity evaluation ends here

        weight_factors = torch.pow(2.0, 10.0 * aux_weight - 5.0)
        aux_loss = torch.mean(train_loss2 * weight_factors)
        train_loss = torch.mean(train_loss1) + aux_loss

        train_loss.backward()

        optimizer.step()

        train_predict_label1 = train_pred1.data.max(1)[1]
        train_acc1 = train_predict_label1.eq(train_label).sum().item() / batch_size

        cost[0] = torch.mean(train_loss1).item()
        cost[1] = train_acc1
        cost[2] = cos_mean
        k = k + 1
        avg_cost[index][0:3] += cost[0:3] / train_batch

    # evaluating training data (meta-training step, update on theta_2)
    cifar100_train_dataset = iter(dataloader_train)
    for i in range(train_batch):
        train_data, train_label = next(cifar100_train_dataset)
        train_label = train_label.type(torch.LongTensor)
        train_data, train_label = train_data.to(device), train_label.to(device)

        train_pred1, train_pred2 = VGG16_model(train_data)
        train_pred3, aux_weight = LabelGenerator(train_data, train_label)

        # reset optimizer with zero gradient
        optimizer.zero_grad()
        gen_optimizer.zero_grad()

        # choose level 2/3 hierarchy, 20-class/100-class classification
        train_loss1 = VGG16_model.model_fit(train_pred1, train_label, pri=True, num_output=PRIMARY_CLASS)
        train_loss2 = VGG16_model.model_fit(train_pred2, train_pred3, pri=False, num_output=AUXILIARY_CLASS)
        train_loss3 = VGG16_model.model_entropy(train_pred3)

        # multi-task loss
        # element-wise multiplication between auxiliary loss and auxiliary weight
        weight_factors = torch.pow(2.0, 10.0 * aux_weight - 5.0)
        aux_loss = torch.mean(train_loss2 * weight_factors)
        train_loss = torch.mean(train_loss1) + aux_loss

        # current accuracy on primary task
        train_predict_label1 = train_pred1.data.max(1)[1]
        train_acc1 = train_predict_label1.eq(train_label).sum().item() / batch_size
        cost[0] = torch.mean(train_loss1).item()
        cost[1] = train_acc1

        # current theta_1
        fast_weights = OrderedDict((name, param) for (name, param) in VGG16_model.named_parameters())

        # create_graph flag for computing second-derivative
        grads = torch.autograd.grad(train_loss, VGG16_model.parameters(), create_graph=True)
        data = [p.data for p in list(VGG16_model.parameters())]

        # compute theta_1^+ by applying sgd on multi-task loss
        fast_weights = OrderedDict((name, param - vgg_lr * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))

        # compute primary loss with the updated thetat_1^+
        train_pred1, train_pred2 = VGG16_model.forward(train_data, fast_weights)
        train_loss1 = VGG16_model.model_fit(train_pred1, train_label, pri=True, num_output=PRIMARY_CLASS)

        # update theta_2 with primary loss + entropy loss
        (torch.mean(train_loss1) + 0.2 * torch.mean(train_loss3)).backward()
        gen_optimizer.step()

        train_predict_label1 = train_pred1.data.max(1)[1]
        train_acc1 = train_predict_label1.eq(train_label).sum().item() / batch_size

        # accuracy on primary task after one update
        cost[2] = torch.mean(train_loss1).item()
        cost[3] = train_acc1
        avg_cost[index][3:7] += cost[0:4] / train_batch

    # evaluate on test data
    VGG16_model.eval()
    with torch.no_grad():
        cifar100_test_dataset = iter(dataloader_test)
        for i in range(test_batch):
            test_data, test_label = next(cifar100_test_dataset)
            test_label = test_label.type(torch.LongTensor)
            test_data, test_label = test_data.to(device), test_label.to(device)
            test_pred1, test_pred2 = VGG16_model(test_data)

            test_loss1 = VGG16_model.model_fit(test_pred1, test_label, pri=True, num_output=PRIMARY_CLASS)

            test_predict_label1 = test_pred1.data.max(1)[1]
            test_acc1 = test_predict_label1.eq(test_label).sum().item() / batch_size

            cost[0] = torch.mean(test_loss1).item()
            cost[1] = test_acc1

            avg_cost[index][7:] += cost[0:2] / test_batch

    scheduler.step()
    gen_scheduler.step()

    epoch_performance = EpochPerformance(
        epoch=index,
        train_loss_primary=avg_cost[index][0],
        train_loss_auxiliary=0,
        train_accuracy_primary=avg_cost[index][1],
        train_accuracy_auxiliary=0,
        test_loss_primary=avg_cost[index][7],
        test_loss_auxiliary=0,
        test_accuracy_primary=avg_cost[index][8],
        test_accuracy_auxiliary=0,
    )
    epoch_performances.append(epoch_performance)

    log_print(epoch_performance)

    # save epoch performances as pickle. has no .save
    with open(save_path + '/epoch_performances.pkl', 'wb') as f:
        pickle.dump(epoch_performances, f)

    log_print('EPOCH: {:04d} Iter {:04d} | TRAIN [LOSS|ACC.]: PRI {:.4f} {:.4f} COSSIM {:.4f} || '
          'META [LOSS|ACC.]: PRE {:.4f} {:.4f} AFTER {:.4f} {:.4f} || TEST: {:.4f} {:.4f}'
          .format(index, k, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2], avg_cost[index][3],
                  avg_cost[index][4], avg_cost[index][5], avg_cost[index][6], avg_cost[index][7], avg_cost[index][8]))