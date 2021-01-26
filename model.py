import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from tensorboardX import SummaryWriter

# upsample from 32x32 to 128x128


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.batch_size = 64
        self.num_epoch = 20

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=2, padding=0)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0)
        self.relu3 = nn.ReLU()

        self.decv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=0)
        self.relu4 = nn.ReLU()

        self.decv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=0)
        self.relu5 = nn.ReLU()

        self.decv3 = nn.ConvTranspose2d(in_channels=512, out_channels=3, kernel_size=4, stride=2, padding=0)
        self.relu6 = nn.ReLU()

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.decv1(x)
        x = self.relu4(x)

        x = self.decv2(x)
        x = self.relu5(x)

        x = self.decv3(x)
        x = self.relu6(x)

        x = self.tanh(x)

        return x
