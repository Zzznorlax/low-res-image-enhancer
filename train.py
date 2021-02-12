
import random

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

import training_utils
import logging_utils
from models import ImageEnhancerNN


def train(device: torch.device, nn_model: nn.Module, optimizer: Optimizer, scale, sample_data):

    sample_data = sample_data.to(device=device)

    randomized_scale = random.uniform(1, scale)

    original_size = ((sample_data.size(2)), sample_data.size(3))
    downscaled_size = (int(sample_data.size(2) / randomized_scale), int(sample_data.size(3) / randomized_scale))

    # Downsamples then upsamples the image so the image's size remains the same,
    # but with resolution lowered
    resized_data = F.interpolate(sample_data, size=downscaled_size)
    resized_data = F.interpolate(resized_data, size=original_size)

    output = nn_model(resized_data)

    # Compares the output of nn_model with original image
    diff = sample_data - output

    # The target is to have minimized difference between the original image and the enhanced image
    # Although it could be a bit of rough, we expect the average of diff to be as close to zero as possible
    average_diff = diff.mean().unsqueeze(-1)

    target = torch.FloatTensor([0])
    target = target.to(device=device)

    loss = criterion(average_diff, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return average_diff


if __name__ == "__main__":

    logger = logging_utils.get_logger()
    logging_utils.add_stdout_handler(logger=logger)

    dataset_path = ""
    model_path = training_utils.DEFAULT_FILE_PATH
    state_path = training_utils.DEFAULT_FILE_PATH
    train_state_file_path = None

    nn_model_name = "image_enhancer"

    num_workers = 0  # Setting to zero to disable multiprocessing
    shuffle = True

    batch_size = 10
    num_epochs = 10

    learning_rate = 2e-4

    scale = 4

    save_period = 50  # Save train state every 50 iterations

    # TODO Set criterion by option
    criterion = nn.L1Loss()

    device = training_utils.setup_device(use_cpu=True)

    train_dataset = training_utils.load_dataset(dataset_path)

    logger.debug("Loading dataset...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )

    logger.debug("Dataset loaded")

    # TODO Set optimizer type by option
    nn_model = ImageEnhancerNN()
    optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate)

    current_epoch = 1
    current_iteration = 1

    # Load state if train state file exists
    if train_state_file_path is not None:
        train_state = training_utils.load_state(model=nn_model, optimizer=optimizer, path=train_state_file_path)

        current_epoch = train_state.epoch
        current_iteration = train_state.iteration

    logger.debug("Training session starting...")

    for epoch in range(current_epoch, num_epochs):
        for batch_idx, (data, label) in enumerate(train_loader):

            average_diff = train(device=device, nn_model=nn_model, optimizer=optimizer, scale=scale, sample_data=data)

            logger.debug("Training... Epoch: ({}/{}), Iteration: {} -> Diff: {}".format(epoch, num_epochs, current_iteration, average_diff[0]))

            current_iteration += 1

            if current_iteration % save_period == 0:
                training_utils.save_model(model=nn_model, name=nn_model_name, epoch=epoch, path=model_path)
                training_utils.save_state(model=nn_model, optimizer=optimizer, name=nn_model_name, epoch=epoch, iteration=current_iteration, path=state_path)
