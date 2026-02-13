import argparse
import torch
import os
from model.smooth_cross_entropy import smooth_crossentropy
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from tools.metric import MomentHubs
from model import ResNet_cifar
from model import Resnet_LT
from model.wide_res_net import WideResNet

import sys; sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import pandas as pd

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



def compare_batch_norm_layers(model1, model2):
    for (name1, layer1), (name2, layer2) in zip(model1.named_children(), model2.named_children()):
        if isinstance(layer1, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)) and isinstance(layer2, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            print(f'Comparing BatchNorm layers {name1} and {name2}:')
            print(f'Weight difference: {torch.equal(layer1.weight, layer2.weight)}')
            print(f'Bias difference: {torch.equal(layer1.bias, layer2.bias)}')
            print(f'Running mean difference: {torch.equal(layer1.running_mean, layer2.running_mean)}')
            print(f'Running variance difference: {torch.equal(layer1.running_var, layer2.running_var)}')
            print('---')


def weight_diff(weights1,weights2):
    # 判断权重是否不同
    weight_difference = False
    for name, param_1 in weights1.items():
        if name in weights2:
            param_2 = weights2[name]
            if not torch.equal(param_1, param_2):
                weight_difference = True
                print(f"Weight difference found for parameter '{name}'")

    if not weight_difference:
        print("No weight difference found between the two checkpoints")

def calculate_direction_and_similarity(net_1, net_2, args):
    # 载入权重数据到模型
    weights1 = net_1.state_dict()
    weights2 = net_2.state_dict()
    
    direction_vector_norms = []
    cosine_values = []

    # 遍历每个层的参数
    keys = []
    for name, param1 in net_1.named_parameters():
        param2 = weights2[name]
        diff = param2 - param1
        keys.append(name)

        # 计算差值方向向量的2-范数
        direction_vector_norms.append(torch.norm(diff, 2).item())

        # 计算权重向量与差值方向向量的余弦相似度
        cosine_value = torch.nn.functional.cosine_similarity(param1.reshape(-1), diff.reshape(-1),dim=0).item()
        cosine_values.append(cosine_value)

    # 绘制柱状图
    plt.figure(figsize=(40, 24))  # 调整图像尺寸为更大的尺寸
    plt.bar(np.arange(len(direction_vector_norms)), direction_vector_norms)
    plt.xticks(np.arange(len(direction_vector_norms)), keys, rotation=90)
    plt.xlabel('layers')
    plt.ylabel('layers2-norm')
    plt.savefig('./checkpoint/{}/{}/{}/direction_vector_norms.png'.format(args.arch, args.dataset, args.exp_name))
    plt.close()

    norms_dict = {}
    norms_dict['name'] = []
    norms_dict['norm'] = []
    for key,value in zip(keys,direction_vector_norms):
        norms_dict['name'].append(key)
        norms_dict['norm'].append(value)
    # 将字典转换为 DataFrame
    df = pd.DataFrame(norms_dict)
    # 将 DataFrame 存储为 CSV 文件
    df.to_csv('./checkpoint/{}/{}/{}/direction_vector_norms.csv'.format(args.arch, args.dataset, args.exp_name), index=False)

    # 绘制热力图
    plt.figure(figsize=(40, 24))  # 调整图像尺寸为更大的尺寸
    plt.bar(np.arange(len(cosine_values)), cosine_values)
    plt.xticks(np.arange(len(cosine_values)), keys, rotation=90)
    plt.xlabel('layers')
    plt.ylabel('cosine ')
    plt.savefig('./checkpoint/{}/{}/{}/direction_vector_cosine.png'.format(args.arch, args.dataset, args.exp_name))
    plt.close()

    cosine_dict = {}
    cosine_dict['name'] = []
    cosine_dict['cosine'] = []
    for key,value in zip(keys,cosine_values):
        cosine_dict['name'].append(key)
        cosine_dict['cosine'].append(value)
        # 将字典转换为 DataFrame
    df = pd.DataFrame(cosine_dict)
    # 将 DataFrame 存储为 CSV 文件
    df.to_csv('./checkpoint/{}/{}/{}/direction_vector_cosine.csv'.format(args.arch, args.dataset, args.exp_name), index=False)



    # 返回结果
    result = {
        'direction_vector_norms': direction_vector_norms,
        'cosine_values': cosine_values
    }
    
    return result

def mkdir(path):
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
        print (path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print (path+' 目录已存在')
        return False

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
    parser.add_argument("--dataset", default="cifar100", type=str, help="dataset")
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')
    parser.add_argument("--checkpoint_start", default=None, type=str, help="checkpoint name")
    parser.add_argument("--checkpoint_end", default=None, type=str, help="checkpoint name")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    if args.checkpoint_start is None or args.checkpoint_end is None:
            raise Exception("Error: need checkpoint_start and checkpoint_end")
    
    model_start = get_model(args)
    if args.checkpoint_start:
        # Load checkpoint.
        print('==> Resuming start from checkpoint:{}/{}/{}'.format(args.arch, args.dataset, args.checkpoint_start))
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}/{}/{}'.format(args.arch, args.dataset, args.checkpoint_start))
        model_start.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    model_end = get_model(args)
    if args.checkpoint_end:
        # Load checkpoint.
        print('==> Resuming end from checkpoint:{}/{}/{}'.format(args.arch, args.dataset, args.checkpoint_end))
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}/{}/{}'.format(args.arch, args.dataset, args.checkpoint_end))
        model_end.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    weights1 = model_start.state_dict().copy()
    weights1_bp = copy.deepcopy(weights1)

    weights2 = model_end.state_dict()

    mkdir("./checkpoint/{}/{}/{}".format(args.arch, args.dataset, args.exp_name))

    calculate_direction_and_similarity(model_start,model_end,args)

    #判断两个网络的bn层的一致性
    compare_batch_norm_layers(model_start,model_end)
    #判断是否有 weight的一致
    weight_diff(weights1,weights2)

    #获得模型的加方向
    direction_vector = {}
    for name, param1 in weights1.items():
        if name in weights2:
            param2 = weights2[name]
            diff = param2 - param1 
            direction_vector[name] = diff
        else:
            print("there {} loss between weight1 and weight 2".format(name))
            #direction_vector[name] = param1

    direction_vector_bp = copy.deepcopy(direction_vector)

    

    # 使用model1+alpha*direction 创建一个新的模型 , model_vector
    for alpha in range(-5, 15, 1):
        alpha = alpha/10.0
        new_state_dict = model_start.state_dict().copy()
        for name, param in new_state_dict.items():
            new_state_dict[name] = weights1_bp[name] + alpha * direction_vector_bp[name]
        model_start.load_state_dict(new_state_dict)
        # Save the entire model
        weight_diff(model_start.state_dict(), weights1_bp)
        torch.save(model_start, './checkpoint/{}/{}/{}/model_alpha_{}.pth'.format(args.arch, args.dataset, args.exp_name,alpha))

    