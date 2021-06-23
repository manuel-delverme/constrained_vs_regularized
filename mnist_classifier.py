from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_constrained
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import config


class MNIST_one_to_five(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        idx_one_to_five = np.where(self.targets < 6)
        self.data = self.data[idx_one_to_five]
        self.targets = self.targets[idx_one_to_five]
        # print("data", self.data)
        # print("targets", self.targets)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index


class MNIST_six(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        idx_six = np.where(self.targets == 6)
        self.data = self.data[idx_six]
        self.targets = self.targets[idx_six]

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # for j, x in enumerate(train_loader):
    #   print(x[0].shape)
    #   print(x[1].shape)
    #   print(x[2])
    #   if j >= 2:
    #     raise SystemExit
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            # print("epoch is", epoch)
            # print("length of train loader is", len(train_loader))
            # print("batch_idx is", batch_idx)
            # print("together they give index", (epoch - 1) * len(train_loader) + batch_idx)

            config.tensorboard.add_scalar("train/loss", loss.item(),
                                          (epoch - 1) * len(train_loader) + batch_idx)
            if args.dry_run:
                break
        # break

    # Return the final loss
    return loss, (epoch - 1) * len(train_loader) + batch_idx


def train_constrained(args, model, device, train_loader_original,
                      train_loader_new, optimizer, epoch, loss_on_previous, previous_idx):
    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader_new):
        # optimizer.zero_grad()

        # Primals
        data, target = data.to(device), target.to(device)

        # Duals (need to load in previous data for that)
        previous_data, previous_target, _ = iter(train_loader_original).next()  # TODO: is this randomized? If not, randomize.

        def closure():
            output = model(data)
            loss = F.nll_loss(output, target)
            dual_output = model(previous_data)
            dual_loss = F.nll_loss(dual_output, previous_target)
            ineq_defect = [(dual_loss - loss_on_previous - 0.025).reshape(1, -1), ]  # max(defect, 0.)
            return loss, None, ineq_defect

        optimizer.step(closure)

        if batch_idx % args.log_interval == 0:
            loss, eq_defect, ineq_defect = closure()
            print('Constrained Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader_new.dataset),
                       100. * batch_idx / len(train_loader_new), loss.item()))
            print('Constrained Ineq Defect Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader_new.dataset),
                       100. * batch_idx / len(train_loader_new), ineq_defect[0].item()))

            # print("epoch is", epoch)
            # print("length of train loader is", len(train_loader_new))
            # print("batch_idx is", batch_idx)
            # print("together they give index", (epoch-1) * len(train_loader_new) + batch_idx)

            config.tensorboard.add_scalar("constrained_train/loss", loss.item(),
                                          (epoch - 1) * len(train_loader_new) + batch_idx + previous_idx)
            for j, defect in enumerate(ineq_defect):
                config.tensorboard.add_scalar("constrained_train/ineq_defect_{}".format(j), defect.item(),
                                              (epoch - 1) * len(train_loader_new) + batch_idx + previous_idx)
            if args.dry_run:
                break


def test(model, device, test_loader, i):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    config.tensorboard.add_scalar("test/loss", test_loss, i)
    config.tensorboard.add_scalar("test/accuracy", correct / len(test_loader.dataset), i)


def main():
    # Training settings
    # config = 1
    import config
    use_cuda = False  # not config.no_cuda and torch.cuda.is_available()

    torch.manual_seed(config.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': config.batch_size}
    test_kwargs = {'batch_size': config.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNIST_one_to_five('./mnist_torchvision/', train=True, download=True, transform=transform)
    # print("we got mnist one to five")
    # raise SystemExit

    # dataset_one_to_five = MNIST_one_to_five('./mnist_torchvision/', train=True, download=True, transform=transform)
    # dataset1 = datasets.MNIST('./mnist_torchvision/', train=True, download=True, transform=transform)
    # dataset_six = None
    # dataset2 = datasets.MNIST('./mnist_torchvision/', train=False, transform=transform)
    # train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    # test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    dataset_one_to_five = MNIST_one_to_five('./mnist_torchvision/', train=True, download=True, transform=transform)
    dataset_six = MNIST_six('./mnist_torchvision/', train=True, download=True, transform=transform)

    loader_15 = torch.utils.data.DataLoader(dataset_one_to_five, **train_kwargs)
    loader_6 = torch.utils.data.DataLoader(dataset_six, **train_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    loss_on_previous = None
    for epoch in range(1, config.epochs + 1):
        # train(config, model, device, train_loader, optimizer, epoch)
        loss_on_previous, last_idx = train(config, model, device, loader_15, optimizer, epoch)
        # test(model, device, test_loader, epoch)
        # scheduler.step()

    # primal_optimizer = optim.Adadelta(model.parameters(), lr=config.lr)
    # dual_optimizer = optim.SGD(model.parameters(), lr=config.lr)
    # constrained_optimizer = ConstrainedOptimizer(primal_optimizer, dual_optimizer)

    constrained_optimizer = torch_constrained.ConstrainedOptimizer(
        torch_constrained.ExtraAdagrad,
        torch_constrained.ExtraSGD,
        lr_x=config.lr * 1e-3,
        lr_y=config.lr * 1e-4,
        primal_parameters=list(model.parameters()),
    )

    for epoch in range(1, config.epochs + 1 + 10):
        print("training constrained on epoch", epoch)
        train_constrained(config, model, device, loader_15, loader_6,
                          constrained_optimizer, epoch, loss_on_previous.item(), last_idx)

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


def plot(loss, model):
    import torchviz
    import os
    torchviz.make_dot(loss, params=dict(model.named_parameters())).render("/tmp/plot.gv")
    os.system("evince /tmp/plot.gv.pdf")


if __name__ == '__main__':
    main()
