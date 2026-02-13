import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utility.cutout import Cutout
from data.cifar_prepare import CIFAR100N,CIFAR10N
import numpy as np

def _init_fn(worker_id):
    np.random.seed(77 + worker_id)

# batch_size, threads, data_type="default"
class Cifar10:
    def __init__(self, args):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


        trainset = CIFAR10N(root='./data', split='train', train_ratio=args.train_val_ratio, trust_ratio=0, download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads, worker_init_fn=_init_fn)
        valset = CIFAR10N(root='./data', split='val', train_ratio=args.train_val_ratio, trust_ratio=0, download=True, transform=test_transform)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
        testset = CIFAR10N(root='./data', split='test', download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

        self.train = trainloader
        self.train_set = trainset
        self.val=valloader
        self.val_set = valset
        self.test = testloader
        self.test_set = testset

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class Cifar100:
    def __init__(self, args):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        trainset = CIFAR100N(root='./data', split='train', train_ratio=args.train_val_ratio, trust_ratio=0, download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads,worker_init_fn=_init_fn)
        valset = CIFAR100N(root='./data', split='val', train_ratio=args.train_val_ratio, trust_ratio=0, download=True, transform=test_transform)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

        testset = CIFAR100N(root='./data', split='test', download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

        self.train = trainloader
        self.train_set = trainset
        self.val=valloader
        self.val_set = valset
        self.test = testloader
        self.test_set = testset

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR100(root='./cifar', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])