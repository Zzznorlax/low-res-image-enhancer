
from typing import Optional

import torch
import torchvision
import sys

from torch import nn
from torch.optim import Optimizer

from models import TrainState

DEVICE_TYPE_CUDA = "cuda:0"
DEVICE_TYPE_CPU = "cpu"

DEFAULT_FILE_PATH = sys.path[0]
MODEL_NAME_FORMAT = "{}_{}.pt"
TRAIN_STATE_FILENAME_FORMAT = "{}_{}_{}.tar"


def setup_device(use_cpu: bool = False) -> torch.device:

    if use_cpu:
        return torch.device(DEVICE_TYPE_CPU)

    device_type = DEVICE_TYPE_CUDA if torch.cuda.is_available() else DEVICE_TYPE_CPU

    return torch.device(device_type)


def load_dataset(path: str):
    train_dataset = torchvision.datasets.ImageFolder(
        root=path,
        transform=torchvision.transforms.ToTensor()
    )
    return train_dataset


def save_model(model: nn.Module, name: str, epoch: int, path: Optional[str] = DEFAULT_FILE_PATH):

    file_path = path + MODEL_NAME_FORMAT.format(name, epoch)

    torch.save(model.state_dict(), file_path)

    print("[INFO] Model saved")


def save_state(model: nn.Module, optimizer: Optimizer, name: str, epoch: int, iteration: int, path: Optional[str] = DEFAULT_FILE_PATH):

    path = path + TRAIN_STATE_FILENAME_FORMAT.format(name, epoch, iteration)

    train_state = TrainState(epoch=epoch, iteration=iteration, model_state=model.state_dict(), optimizer_state=optimizer.state_dict())

    torch.save(
        vars(train_state),
        path,
    )


def load_state(model: nn.Module, optimizer: Optimizer, path: Optional[str] = DEFAULT_FILE_PATH) -> TrainState:
    data = torch.load(path)

    train_state = TrainState(**data)

    model.load_state_dict(train_state.model_state)
    optimizer.load_state_dict(train_state.optimizer_state)

    return train_state


# def train(model, optimizer, data_path, batch_size, num_epochs, curr_epoch, curr_iter, writer, model_path, state_path):
#     print("[INFO] Start training session")

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("[INFO] Device:", device)

#     print("[INFO] Loading dataset...")
#     train_dataset = load_dataset(data_path)
#     torch.
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         num_workers=0,
#         shuffle=True
#     )
#     print("[INFO] Finished loading dataset")

#     criterion = nn.L1Loss()

#     print("[INFO] Start training")
#     for epoch in range(curr_epoch + 1, curr_epoch + num_epochs + 1):
#         print('[ITER] Starting epoch:', "[" + str(curr_epoch + 1) + "/" + str(curr_epoch + num_epochs) + "]")
#         for batch_idx, (data, label) in enumerate(train_loader):
#             data = data.to(device=device)

#             scale = random.uniform(1, 4)
#             resized_data = F.interpolate(data, int(data.size(2) / scale))
#             resized_data = F.interpolate(resized_data, data.size(2))

#             output = model(resized_data)

#             diff = data - output
#             diff = diff.mean().unsqueeze(-1)
#             # print("[DEBUG] diff size:", diff.size())

#             target = torch.FloatTensor([0])
#             target = target.to(device=device)
#             # print("[DEBUG] target size:", target.size())

#             loss = criterion(diff, target)

#             writer.add_scalar("loss", loss.item(), curr_iter)

#             print('[ITER] Iteration:', curr_iter)
#             print('[LOSS] Loss:', loss.item())

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             curr_iter += 1

#             if curr_iter % 50 == 0:
#                 save_model(model, model_path, curr_epoch)
#                 save_state(model, epoch, curr_iter, optimizer, state_path)


# def train_from_start(model, data_path, batch_size, num_epochs, writer):
#     optimizer = optim.Adam(model.parameters(), lr=2e-4)
#     train(model, optimizer, data_path, batch_size, num_epochs, 0, 0, writer)


# def train_from_checkpoint(model, state_path, data_path, batch_size, num_epochs, writer):
#     optimizer = optim.Adam(model.parameters(), lr=2e-4)
#     curr_epoch, curr_iter = load_state(model, optimizer, state_path)
#     train(model, optimizer, data_path, batch_size, num_epochs, curr_epoch, curr_iter, writer)
