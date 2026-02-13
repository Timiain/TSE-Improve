import argparse
import torch
import os

from model.smooth_cross_entropy import smooth_crossentropy
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from tools.metric import MomentHubs

from data.cifarn import Cifar100, Cifar10
from utility.noise import noisify_with_P, noisify_cifar10_asymmetric, \
    noisify_cifar100_asymmetric, noisify_mnist_asymmetric, noisify_pairflip, noisify_modelnet40_asymmetric

from model import ResNet_cifar
from model import Resnet_LT
from model import vit
from model.wide_res_net import WideResNet
from model.PyramidNet import PyramidNet

import sys; sys.path.append("..")
from sam import SAM
import numpy as np
#parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
#parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")

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

random_seed=42
def generate_noise(args, trainset,valset,testset):
    arg_dataset = args.dataset
    noise_type = args.noise_type
    noise_level = args.noise_level
    num_class = args.num_classes
    print('train data size:', len(trainset))
    print('validation data size:', len(valset))
    print('test data size:', len(testset))

    eps = 1e-6               # This is the epsilon used to soft the label (not the epsilon in the paper)
    ntrain = len(trainset)

    # -- generate noise --
    # y_train is ground truth labels, we should not have any access to  this after the noisy labels are generated
    # algorithm after y_tilde is generated has nothing to do with y_train
    y_train = trainset.get_data_labels()
    y_train = np.array(y_train)

    noise_y_train = None
    keep_indices = None
    p = None

    if(noise_type == 'none'):
        pass
    else:
        if noise_type == "uniform":
            noise_y_train, p, keep_indices = noisify_with_P(y_train, nb_classes=num_class, noise=noise_level, random_state=random_seed)
            trainset.update_corrupted_label(noise_y_train)
            noise_softlabel = torch.ones(ntrain, num_class)*eps/(num_class-1)
            noise_softlabel.scatter_(1, torch.tensor(noise_y_train.reshape(-1, 1)), 1-eps)
            trainset.update_corrupted_softlabel(noise_softlabel)

            print("apply uniform noise")
        else:
            if arg_dataset == 'cifar10':
                noise_y_train, p, keep_indices = noisify_cifar10_asymmetric(y_train, noise=noise_level, random_state=random_seed)
            elif arg_dataset == 'cifar100':
                noise_y_train, p, keep_indices = noisify_cifar100_asymmetric(y_train, noise=noise_level, random_state=random_seed)
            # elif arg_dataset == 'mnist':
            #     noise_y_train, p, keep_indices = noisify_mnist_asymmetric(y_train, noise=noise_level, random_state=random_seed)
            # elif arg_dataset == 'pc':
            #     noise_y_train, p, keep_indices = noisify_modelnet40_asymmetric(y_train, noise=noise_level,
            #                                                                    random_state=random_seed)
            trainset.update_corrupted_label(noise_y_train)
            noise_softlabel = torch.ones(ntrain, num_class) * eps / (num_class - 1)
            noise_softlabel.scatter_(1, torch.tensor(noise_y_train.reshape(-1, 1)), 1 - eps)
            trainset.update_corrupted_softlabel(noise_softlabel)

            print("apply asymmetric noise")
        print("clean data num:", len(keep_indices))
        print("probability transition matrix:\n{}".format(p))
    return trainset, y_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--cuda", default=0, type=int, help="gpu index")
    parser.add_argument("--exp_name", default="default", type=str, help="exp_name")
    parser.add_argument("--arch", default="wrn28-10", type=str, help="model_name")
    parser.add_argument("--dataset", default="cifar100n", type=str, help="dataset")
    parser.add_argument("--noise_type", default="asymmetric", type=str, help="noise_type")
    parser.add_argument("--noise_level", default=0.2, type=float, help="train_val_ratio to split dataset.")
    parser.add_argument("--train_val_ratio", default=1.0, type=float, help="train_val_ratio to split dataset.")
    parser.add_argument("--checkpoint", default=None, type=str, help="checkpoint name")
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    best_acc = 0
    log = Log(log_each=10)

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir('checkpoint/{}'.format(args.arch)):
        os.mkdir('checkpoint/{}'.format(args.arch))
    if not os.path.isdir('checkpoint/{}/{}'.format(args.arch,args.dataset)):
        os.mkdir('checkpoint/{}/{}'.format(args.arch,args.dataset))

    if args.dataset == 'cifar10n':
        print("Using data set cifar10N..")
        dataset = Cifar10(args)
    if args.dataset == 'cifar100n':
        print("Using data set cifar100N..")
        dataset = Cifar100(args)

    # add noise
    train_set,y_train = generate_noise(args,dataset.train_set,dataset.val_set,dataset.test_set)
    dataset.train_set=train_set

    model = get_model(args)
    
    if args.checkpoint:
        # Load checkpoint.
        print('==> Resuming from checkpoint:{}/{}/{}'.format(args.arch, args.dataset, args.checkpoint))
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}/{}/{}'.format(args.arch, args.dataset, args.checkpoint))
        model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    train_recoder = MomentHubs()
    val_recoder = MomentHubs()

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            #(images, labels, softlabels, indices)
            inputs, targets, softlabels, indices = (b.to(device) for b in batch)

            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            # optimizer.first_step(zero_grad=True)

            # # second forward-backward step
            # disable_running_stats(model)
            # smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            # optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)
            train_recoder.append("loss",loss.mean().cpu().item())

        model.eval()
        log.eval(len_dataset=len(dataset.test),need_flush=True)

        with torch.no_grad():
            _correct = 0
            total = 0
            for batch in dataset.test:
                #(images, labels, softlabels, indices)
                inputs, targets, softlabels, indices = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
                #为了作区分，前面有个tensor correct
                total += targets.size(0)
                _correct += torch.sum(correct).detach().cpu().item()
                val_recoder.append("loss",loss.mean().cpu().item())

            # Save checkpoint.
            acc = 100.*_correct/total
            if  acc > best_acc:
                print('Saving..')
                state = {
                    'net': model.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state, './checkpoint/{}/{}/{}.pth'.format(args.arch,args.dataset,args.exp_name))
                best_acc = acc
        train_recoder.step_summary()
        val_recoder.step_summary()

    train_recoder.to_csv("train_{}_{}_{}.csv".format(args.arch,args.dataset,args.exp_name))
    val_recoder.to_csv("test_{}_{}_{}.csv".format(args.arch,args.dataset,args.exp_name))

    log.flush()
