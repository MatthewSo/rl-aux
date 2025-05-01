import torch
import torch.nn as nn

# Define VGG net
class VGG16(nn.Module):
    def __init__(self, primary_task_output, auxiliary_task_output, input_shape=(3,32,32)):
        super(VGG16, self).__init__()
        """
            multi-task network:
            takes the input and predicts primary and auxiliary labels (same network structure as in human)
        """
        self._pri_out = primary_task_output
        self._aux_out = auxiliary_task_output

        filter = [64, 128, 256, 512, 512]

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = self._forward_conv(dummy)
            flat_dim = dummy.view(1, -1).size(1)

        # define convolution block in VGG-16
        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        # primary task prediction
        self.classifier1 = nn.Sequential(
            nn.Linear(flat_dim, filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], primary_task_output),
            nn.Softmax(dim=1),
        )

        # auxiliary task prediction
        self.classifier2 = nn.Sequential(
            nn.Linear(flat_dim, filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], auxiliary_task_output),
            nn.Softmax(dim=1),
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

    def _forward_conv(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

    def forward(self, x):
        g_block1 = self.block1(x)
        g_block2 = self.block2(g_block1)
        g_block3 = self.block3(g_block2)
        g_block4 = self.block4(g_block3)
        g_block5 = self.block5(g_block4)

        t1_pred = self.classifier1(g_block5.view(g_block5.size(0), -1))
        t2_pred = self.classifier2(g_block5.view(g_block5.size(0), -1))
        return t1_pred, t2_pred

    def model_fit(self, x_pred, x_output, device, pri=True, num_output=3):
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

    def save(self, path):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "primary_task_output": self._pri_out,
                "auxiliary_task_output": self._aux_out,
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        model = cls(ckpt["primary_task_output"], ckpt["auxiliary_task_output"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model