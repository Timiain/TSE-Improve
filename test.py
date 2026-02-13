import argparse
import torch
import os
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from tools.metric import MomentHubs

from data.cifar import Cifar100, Cifar
from model import ResNet_cifar
from model.PyramidNet import PyramidNet

import sys; sys.path.append("..")
from sam import SAM

def get_model(args):
    if args.arch == 'wrn16_8':
        print("Using arch wrn16_8..")
        return WideResNet(16, 8, args.dropout, in_channels=3, labels=args.num_classes).to(device)
    if args.arch == 'wrn28_2':
        print("Using arch wrn28_2..")
        return WideResNet(28, 2, args.dropout, in_channels=3, labels=args.num_classes).to(device)
    if args.arch == 'wrn28_10':
        print("Using arch wrn28_10..")
        return WideResNet(28, 10, args.dropout, in_channels=3, labels=args.num_classes).to(device)
    if args.arch == 'resnet_50':
        print("Using arch resnet_50..")
        return ResNet_cifar.resnet50(num_class=args.num_classes).to(device)
    if args.arch == 'resnet_18':
        print("Using arch resnet_18..")
        return ResNet_cifar.resnet18(num_class=args.num_classes).to(device)
    if args.arch == 'vit_b_4':
        print("Using arch vit_b_4..")
        return vit.ViT_B_4(num_classes=args.num_classes).to(device)
    if args.arch == 'pyramidnet_272':
        print("Using arch pyramidnet_272..")
        bottleneck = True
        alpha = 24
        depth = 272
        return PyramidNet(args.dataset, depth, alpha, args.num_classes, bottleneck) # for ResNet  
    if args.arch == 'pyramidnet_110':
        print("Using arch pyramidnet_110..")
        bottleneck = True
        alpha = 48
        depth = 110
        return PyramidNet(args.dataset, depth, alpha, args.num_classes, bottleneck).to(device) # for ResNet  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--cuda", default=0, type=int, help="gpu index")
    parser.add_argument("--exp_name", default="default", type=str, help="exp_name")
    parser.add_argument("--arch", default="wrn28-10", type=str, help="model_name")
    parser.add_argument("--dataset", default="cifar100", type=str, help="dataset")
    parser.add_argument("--checkpoint", default=None, type=str, help="checkpoint name")
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')
    parser.add_argument("--eval_train", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    best_acc = 0
    log = Log(log_each=10)

    if args.dataset == 'cifar10':
        print("Using data set cifar10..")
        dataset = Cifar(args.batch_size, args.threads)
    if args.dataset == 'cifar100':
        print("Using data set cifar100..")
        dataset = Cifar100(args.batch_size, args.threads)

    model = get_model(args)

    if args.checkpoint:
        # Load checkpoint.
        print('==> Resuming from checkpoint:{}/{}/{}'.format(args.arch, args.dataset, args.checkpoint))
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}/{}/{}'.format(args.arch, args.dataset, args.checkpoint))
        model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    
    if args.eval_train:
        print("trainset #------------------------")
        model.eval()
        log.eval(len_dataset=len(dataset.train),need_flush=False)

        with torch.no_grad():
            _correct = 0
            total = 0
            for batch in dataset.train:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
                #为了作区分，前面有个tensor correct
                total += targets.size(0)
                _correct += torch.sum(correct).detach().cpu().item()
        log.flush()


    print("testset #------------------------")
    model.eval()
    log.eval(len_dataset=len(dataset.test),need_flush=False)

    with torch.no_grad():
        _correct = 0
        total = 0
        for batch in dataset.test:
            inputs, targets = (b.to(device) for b in batch)

            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            correct = torch.argmax(predictions, 1) == targets
            log(model, loss.cpu(), correct.cpu())
            #为了作区分，前面有个tensor correct
            total += targets.size(0)
            _correct += torch.sum(correct).detach().cpu().item()
    log.flush()


    

