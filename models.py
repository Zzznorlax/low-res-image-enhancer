from typing import Dict

import torch.nn as nn


class ImageEnhancerNN(nn.Module):
    def __init__(self):
        super(ImageEnhancerNN, self).__init__()

        self.convolution_layer_1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=2, padding=0)
        self.relu_layer_1 = nn.ReLU()

        self.convolution_layer_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0)
        self.relu_layer_2 = nn.ReLU()

        self.convolution_layer_3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0)
        self.relu_layer_3 = nn.ReLU()

        self.deconvolution_layer_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=0)
        self.relu_layer_4 = nn.ReLU()

        self.deconvolution_layer_2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=0)
        self.relu_layer_5 = nn.ReLU()

        self.deconvolution_layer_3 = nn.ConvTranspose2d(in_channels=512, out_channels=3, kernel_size=4, stride=2, padding=0)
        self.relu_layer_6 = nn.ReLU()

        self.tanh_layer = nn.Tanh()

    def forward(self, x):
        x = self.convolution_layer_1(x)
        x = self.relu_layer_1(x)

        x = self.convolution_layer_2(x)
        x = self.relu_layer_2(x)

        x = self.convolution_layer_3(x)
        x = self.relu_layer_3(x)

        x = self.deconvolution_layer_1(x)
        x = self.relu_layer_4(x)

        x = self.deconvolution_layer_2(x)
        x = self.relu_layer_5(x)

        x = self.deconvolution_layer_3(x)
        x = self.relu_layer_6(x)

        x = self.tanh_layer(x)

        return x


class TrainState():

    def __init__(self, epoch: int, iteration: int, model_state: Dict, optimizer_state: Dict) -> None:
        self.epoch = epoch
        self.iteration = iteration
        self.model_state = model_state
        self.optimizer_state = optimizer_state
