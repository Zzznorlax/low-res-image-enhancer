import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import os
import sys
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


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


def preprocess_img(img_path):
    """Loads image with size=28 * 28, then converts it into nn readable tensor"""
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = TF.to_tensor(img)
    img.unsqueeze_(0)

    return img


def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def imshow(tensor, title=None):
    unloader = torchvision.transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(1000)  # pause a bit so that plots are updated


def show_multiple_imgs(imgs):
    fig = plt.figure(figsize=(20, 20))
    plt.tight_layout(True)
    ax = []
    cols = len(imgs)
    rows = len(imgs[0])
    print("[INFO] Received imgs:", cols, rows)

    i = 0
    for row in range(rows):
        for col in range(cols):
            if i - cols < 0:
                ax.append(fig.add_subplot(rows, cols, i + 1))
            else:
                ax.append(fig.add_subplot(rows, cols, i + 1, sharex=ax[i - cols], sharey=ax[i - cols]))

            ax[-1].set_title("ax:" + str(i))  # set title
            plt.imshow(imgs[col][row])
            i += 1
    plt.pause(1000)
