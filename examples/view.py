#!/usr/bin/env python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


def main():
    ds = datasets.MNIST(root="data", download=True, transform=ToTensor())
    print("images: {} {}".format(ds.data.size(), ds.data.dtype))
    print("labels: {} {} classes={}".format(ds.targets.size(), ds.targets.dtype, len(ds.classes)))

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        idx = torch.randint(len(ds), size=(1,)).item()
        img, label = ds[idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

if __name__ == '__main__':
    main()