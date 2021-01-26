import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


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
