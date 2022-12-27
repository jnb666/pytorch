#!/usr/bin/env python
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import argparse

def mlp():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

def get_data(train, device='cpu'):
    ds = datasets.MNIST(root='/home/john/pytorch/data', train=train, download=True, transform=ToTensor())
    ds.data = ds.data.type(torch.FloatTensor) / 255.0
    ds.data, ds.targets = ds.data.to(device), ds.targets.to(device)
    name = 'Train' if train else 'Test '
    print('{} images: {} {}'.format(name, ds.data.size(), ds.data.dtype))
    print('{} labels: {} {} classes={}'.format(name, ds.targets.size(), ds.targets.dtype, len(ds.classes)))
    return ds.data, ds.targets

def train(trainX, trainY, model, optimizer, loss_fn):
    model.train()
    pred = model(trainX)
    loss = loss_fn(pred, trainY)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def test(testX, testY, model):
    model.eval()
    with torch.no_grad():
        pred = model(testX).argmax(1)
        correct = (pred == testY).type(torch.float).sum().item()
    return int(correct)

def main(args):
    device = 'cuda' if not args.cpu and torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    print('Using {} device  seed = {}'.format(device, args.seed))

    trainX, trainY = get_data(True, device)
    testX, testY = get_data(False, device)
    ntest = len(testY)

    model = mlp().to(device)
    print('Model: {}'.format(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print('Optimizer: {}'.format(optimizer))
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        loss = train(trainX, trainY, model, optimizer, loss_fn)
        correct = test(testX, testY, model)
        if (epoch+1) % 10 == 0:
            print('Epoch {:3d}:  Loss: {:.3f}  Accuracy: {}/{} {:.2f}%'.format(
                epoch+1, loss, correct, ntest, 100*(correct/ntest)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--cpu', action='store_true', default=False, help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    main(args)