import warnings
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.datasets import CIFAR10, CIFAR100


def get_data(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar100':
        train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'mnist':
        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform_train)
        val_dataset = MNIST(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'fmnist':
        train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
        val_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader
