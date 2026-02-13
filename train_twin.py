import argparse
import torch
import os
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from tools.metric import MomentHubs
from data.cifar import Cifar100, Cifar

from model import ResNet_cifar
from model import Resnet_LT
from model import vit
from model.PyramidNet import PyramidNet
from model.wide_res_net import WideResNet
import torch.nn as nn
import torch.optim as optim
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


# 计算Fisher信息矩阵
def compute_fisher_information(model, dataset, device):
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    fisher_information = {}
    for batch in dataset.train:
        inputs, targets = (b.to(device) for b in batch)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # 计算梯度的平方并加到对应参数的fisher_information中
        for name, param in model.named_parameters():
            #print(name)
            fisher_information[name] = fisher_information.get(name, 0) + param.grad.data ** 2 / len(dataset.train)
    return fisher_information

# 损失函数中加入EWC的正则化项
def ewc_loss(model, importance, fisher_information, params):
    ewc_term = 0
    for name, param in model.named_parameters():
        if name in fisher_information:
            ewc_term += torch.sum(fisher_information[name] * (param - params[name]) ** 2) * importance
    return ewc_term

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
    parser.add_argument("--checkpoint", default=None, type=str, help="a name for initial point ")
    parser.add_argument('--lamda', type=float, default=400)
    parser.add_argument("--checkpoint_parentA", default=None, type=str, help="a name for source A")
    parser.add_argument("--checkpoint_parentB", default=None, type=str, help="a name for source B")
    parser.add_argument("--alpha", type=float, default=0.9, help="control the relationship between A and B ")
    parser.add_argument('--online', action='store_true', default=False)
    parser.add_argument("--arch", default="wrn28-10", type=str, help="model_name")
    parser.add_argument("--dataset", default="cifar10", type=str, help="dataset")
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    best_acc = 0

    if args.dataset == 'cifar10':
        print("Using data set cifar10..")
        dataset = Cifar(args.batch_size, args.threads)
    if args.dataset == 'cifar100':
        print("Using data set cifar100..")
        dataset = Cifar100(args.batch_size, args.threads)
    log = Log(log_each=10)
    modelA = get_model(args)
    modelB = get_model(args)
    #model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    if args.checkpoint is None or args.checkpoint_parentA is None or args.checkpoint_parentB is None:
        raise Exception("Error: need checkpoint and checkpoint_parentA and checkpoint_parentB")

    if args.checkpoint_parentA:
        # Load checkpoint.
        print('==> Resuming from checkpointA:{}/{}/{}'.format(args.arch, args.dataset, args.checkpoint_parentA))
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}/{}/{}'.format(args.arch, args.dataset, args.checkpoint_parentA))
        modelA.load_state_dict(checkpoint['net'])

    if args.checkpoint_parentB:
        # Load checkpoint.
        print('==> Resuming from checkpointB:{}/{}/{}'.format(args.arch, args.dataset, args.checkpoint_parentB))
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}/{}/{}'.format(args.arch, args.dataset, args.checkpoint_parentB))
        modelB.load_state_dict(checkpoint['net'])

    # 保存模型参数
    paramsA = {name: param for name, param in modelA.named_parameters()}
    fisher_information_A = compute_fisher_information(modelA, dataset,device)

    paramsB = {name: param for name, param in modelB.named_parameters()}
    fisher_information_B = compute_fisher_information(modelB, dataset,device)


    if args.checkpoint:
        print('==> Resuming from checkpoint:/{}/{}/{}'.format(args.arch, args.dataset, args.checkpoint))
        model_path = './checkpoint/{}/{}/{}'.format(args.arch, args.dataset, args.checkpoint)
        # Load the model
        model = torch.load(model_path)
        model.to(device)

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    train_recoder = MomentHubs()
    val_recoder = MomentHubs()

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss_ce = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss_ewc_a = ewc_loss(model, args.lamda, fisher_information_A, paramsA)  # 添加EWC正则化
            loss_ewc_b = ewc_loss(model, args.lamda, fisher_information_B, paramsB)  # 添加EWC正则化
            loss_ewc = loss_ewc_a*args.alpha+loss_ewc_b*(1-args.alpha)
            loss = loss_ce.mean() + loss_ewc
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss_ce.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)
            train_recoder.append("loss_ce",loss_ce.mean().cpu().item())
            train_recoder.append("loss_ewc_a",loss_ewc_a.mean().cpu().item())
            train_recoder.append("loss_ewc_b",loss_ewc_b.mean().cpu().item())

        model.eval()
        log.eval(len_dataset=len(dataset.test),need_flush=True)

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
