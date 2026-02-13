import argparse
import torch
import os
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar, Cifar100
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from tools.metric import MomentHubs
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import pandas as pd

import sys; sys.path.append("..")
from sam import SAM

def plot_curves(train_file, val_file, save_path):
    # 读取训练数据和验证数据
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)

    # 按alpha列进行排序
    train_data = train_data.sort_values('alpha')
    val_data = val_data.sort_values('alpha')

    print(train_data)
    print(val_data)


    # 创建一个新的图形和轴对象
    fig, ax1 = plt.subplots()

    # 绘制 val_ce_mean 曲线（左坐标轴）
    ax1.plot(train_data['alpha'].values, train_data['loss'].values, linestyle='-', color='blue', label='train loss')
    ax1.plot(val_data['alpha'].values, val_data['loss'].values, linestyle='-', color='red', label='val loss')
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('val_ce_mean')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    # 创建第二个坐标轴对象
    ax2 = ax1.twinx()

    # 绘制 iou 曲线（右坐标轴）
    ax2.plot(train_data['alpha'].values, train_data['acc'].values, linestyle='--', color='blue', label='train acc')
    ax2.plot(val_data['alpha'].values, val_data['acc'].values, linestyle='--', color='red', label='val acc')
    ax2.set_ylabel('acc')
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right')

    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines + lines2, labels + labels2, loc='upper right')


    # 在 alpha=0 和 alpha=1 处添加垂直黑色虚线
    ax1.axvline(x=0, linestyle='--', color='black', alpha=0.5)
    ax1.axvline(x=1, linestyle='--', color='black', alpha=0.5)

    # 标题和图例
    plt.title('Curves')


    # 保存图像
    plt.savefig(save_path)
    plt.close()

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
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')  
    parser.add_argument("--checkpoint", default=None, type=str, help="checkpoint name")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    if args.dataset == 'cifar10':
        print("Using data set cifar10..")
        dataset = Cifar(args.batch_size, args.threads)
    if args.dataset == 'cifar100':
        print("Using data set cifar100..")
        dataset = Cifar100(args.batch_size, args.threads)
    log = Log(log_each=10)
    
    val_recoder_testset=MomentHubs()
    val_recoder_trainset=MomentHubs()
    val_recoder_trainmixset=MomentHubs()

    for alpha in range(-5, 15, 1):
        alpha = alpha/10.0
        # Specify the path to the saved model file
        model_path = './checkpoint/{}/{}/{}/model_alpha_{}.pth'.format(args.arch, args.dataset, args.exp_name,alpha)

        # Load the model
        model = torch.load(model_path).to(device)
        # model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)
        #model.cuda()
        model.eval()
        #model.decoder.val_need_feature=False
        print("test set in alpha={}".format(alpha))
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
        
        loss = log.epoch_state["loss"] / log.epoch_state["steps"]
        accuracy = log.epoch_state["accuracy"] / log.epoch_state["steps"]
        val_recoder_testset.append_direct_to_recoder('alpha',alpha)
        val_recoder_testset.append_direct_to_recoder('loss',loss)
        val_recoder_testset.append_direct_to_recoder('acc',accuracy)
        log.flush()

        print("train set in alpha={}".format(alpha))
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
        loss = log.epoch_state["loss"] / log.epoch_state["steps"]
        accuracy = log.epoch_state["accuracy"] / log.epoch_state["steps"]
        val_recoder_trainset.append_direct_to_recoder('alpha',alpha)
        val_recoder_trainset.append_direct_to_recoder('loss',loss)
        val_recoder_trainset.append_direct_to_recoder('acc',accuracy)
        log.flush()


        # 将模型移动到CPU
        model = model.to('cpu')
        # 清空显存
        torch.cuda.empty_cache()
        del model
        # print("train mix set in alpha={}".format(alpha))
        # trainmix_ret = test(args,config,alpha,model,test_dataset = config.no_argue_train_mix_dataset, val_recoder=val_recoder_trainmixset)
    
    val_recoder_testset.to_csv('./checkpoint/{}/{}/{}/landscape_group_val.csv'.format(args.arch, args.dataset,args.exp_name))
    val_recoder_trainset.to_csv('./checkpoint/{}/{}/{}/landscape_group_train.csv'.format(args.arch, args.dataset,args.exp_name))
    #val_recoder_trainmixset.to_csv('{}_landscape_group_train_mix.csv'.format(config.out_csv))

    plot_curves('./checkpoint/{}/{}/{}/landscape_group_train.csv'.format(args.arch, args.dataset,args.exp_name),
                './checkpoint/{}/{}/{}/landscape_group_val.csv'.format(args.arch, args.dataset,args.exp_name),
                './checkpoint/{}/{}/{}/loss_curves.png'.format(args.arch, args.dataset,args.exp_name))
    exit(0)

    


        

    


    

