#!/usr/bin/env python
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import argparse

def cnn():
     return nn.Sequential(
        nn.Unflatten(0, (-1, 1)),
        nn.Conv2d(1, 32, 5),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 5),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 10),
    )

def get_data(train, device='cpu'):
    ds = datasets.MNIST(root='/home/john/pytorch/data', train=train, download=True, transform=ToTensor())
    ds.data = ds.data.type(torch.FloatTensor) / 255.0
    ds.data, ds.targets = ds.data.to(device), ds.targets.to(device)
    name = 'Train' if train else 'Test '
    print('{} images: {} {}'.format(name, ds.data.size(), ds.data.dtype))
    print('{} labels: {} {} classes={}'.format(name, ds.targets.size(), ds.targets.dtype, len(ds.classes)))
    return ds

def train(ds, model, optimizer, loss_fn, batch_size, device):
    model.train()
    samples = len(ds.targets)
    batches = samples // batch_size
    ix = torch.randperm(samples, device=device)
    images = ds.data.index_select(0, ix)
    labels = ds.targets.index_select(0, ix)
    average_loss = 0
    for i in range(batches):
        data = images[i*batch_size : (i+1)*batch_size]
        target = labels[i*batch_size : (i+1)*batch_size]
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        average_loss += loss.item() / batches
    return average_loss

def test(ds, model, device):
    model.eval()
    with torch.no_grad():
        correct = (model(ds.data).argmax(1) == ds.targets).type(torch.float).sum()
    return int(correct)

def main(args):
    device = 'cuda' if not args.cpu and torch.cuda.is_available() else 'cpu'
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print('Using {} device  seed = {}'.format(device, args.seed))

    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_set = get_data(True, device)
    test_set = get_data(False, device)
    ntest = len(test_set.targets)

    model = cnn().to(device)
    print("Model: {}".format(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    print("Optimizer: {}".format(optimizer))
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        loss = train(train_set, model, optimizer, loss_fn, args.batch, device)
        correct = test(test_set, model, device)
        print('Epoch {:3d}:  Loss: {:.3f}  Accuracy: {}/{} {:.2f}%'.format(
            epoch, loss, correct, ntest, 100*(correct/ntest)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', default=False, help='disable CUDA training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--batch', type=int, default=256, help='training batch size')
    args = parser.parse_args()
    main(args)