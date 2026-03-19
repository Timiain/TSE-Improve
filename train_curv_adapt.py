import argparse
import os

import torch
from torch.utils.data import DataLoader

from data.cifar import Cifar, Cifar100
from model import ResNet_cifar
from model import vit
from model.PyramidNet import PyramidNet
from model.smooth_cross_entropy import smooth_crossentropy
from model.wide_res_net import WideResNet
from sam import CurvAdaptiveSAM
from tools.metric import MomentHubs
from utility.bypass_bn import disable_running_stats, enable_running_stats
from utility.initialize import initialize
from utility.log import Log


def get_model(args, device):
    if args.arch == "wrn16_8":
        print("Using arch wrn16_8..")
        return WideResNet(16, 8, args.dropout, in_channels=3, labels=args.num_classes).to(device)
    if args.arch == "wrn28_2":
        print("Using arch wrn28_2..")
        return WideResNet(28, 2, args.dropout, in_channels=3, labels=args.num_classes).to(device)
    if args.arch == "wrn28_10":
        print("Using arch wrn28_10..")
        return WideResNet(28, 10, args.dropout, in_channels=3, labels=args.num_classes).to(device)
    if args.arch == "resnet_50":
        print("Using arch resnet_50..")
        return ResNet_cifar.resnet50(num_class=args.num_classes).to(device)
    if args.arch == "resnet_18":
        print("Using arch resnet_18..")
        return ResNet_cifar.resnet18(num_class=args.num_classes).to(device)
    if args.arch == "vit_b_4":
        print("Using arch vit_b_4..")
        return vit.ViT_B_4(num_classes=args.num_classes).to(device)
    if args.arch == "pyramidnet_272":
        print("Using arch pyramidnet_272..")
        return PyramidNet(args.dataset, 272, 24, args.num_classes, True).to(device)
    if args.arch == "pyramidnet_110":
        print("Using arch pyramidnet_110..")
        return PyramidNet(args.dataset, 110, 48, args.num_classes, True).to(device)
    raise ValueError(f"Unsupported arch: {args.arch}")


def rebuild_train_loader(dataset, batch_size, threads):
    dataset.train = DataLoader(dataset.train.dataset, batch_size=batch_size, shuffle=True, num_workers=threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True to use Adaptive-SAM style perturbation scaling.")
    parser.add_argument("--batch_size", default=128, type=int, help="Base batch size.")
    parser.add_argument("--min_batch_size", default=32, type=int, help="Minimum adaptive batch size.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum.")
    parser.add_argument("--threads", default=2, type=int, help="DataLoader workers.")
    parser.add_argument("--rho", default=0.5, type=float, help="Perturbation radius for SAM step.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--cuda", default=0, type=int, help="GPU index.")
    parser.add_argument("--exp_name", default="curv_adapt", type=str, help="Experiment name.")
    parser.add_argument("--arch", default="wrn28_10", type=str, help="Model architecture.")
    parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset.")
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint file name.")
    parser.add_argument("--num_classes", default=10, type=int, help="Number of classes.")

    parser.add_argument("--gp_rank", default=32, type=int, help="Random feature rank for surrogate GP sketch.")
    parser.add_argument("--gp_window", default=20, type=int, help="History window for GP sketch updates.")
    parser.add_argument("--gp_update_interval", default=10, type=int, help="Update GP surrogate every K optimizer steps.")
    parser.add_argument("--gp_weight_temperature", default=5.0, type=float, help="Weighting temperature for surrogate update.")
    parser.add_argument("--hessian_momentum", default=0.9, type=float, help="EMA factor for curvature diagonal estimation.")
    parser.add_argument("--curvature_floor", default=1e-6, type=float, help="Minimum curvature value for PSD projection.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    best_acc = 0.0
    log = Log(log_each=10)

    os.makedirs(f"checkpoint/{args.arch}/{args.dataset}", exist_ok=True)

    if args.dataset == "cifar10":
        print("Using data set cifar10..")
        dataset = Cifar(args.batch_size, args.threads)
    elif args.dataset == "cifar100":
        print("Using data set cifar100..")
        dataset = Cifar100(args.batch_size, args.threads)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    model = get_model(args, device)

    if args.checkpoint:
        print(f"==> Resuming from checkpoint:{args.arch}/{args.dataset}/{args.checkpoint}")
        checkpoint = torch.load(f"./checkpoint/{args.arch}/{args.dataset}/{args.checkpoint}")
        model.load_state_dict(checkpoint["net"])
        best_acc = checkpoint["acc"]

    base_optimizer = torch.optim.SGD
    optimizer = CurvAdaptiveSAM(
        model.parameters(),
        base_optimizer,
        rho=args.rho,
        adaptive=args.adaptive,
        base_batch_size=args.batch_size,
        min_batch_size=args.min_batch_size,
        hessian_momentum=args.hessian_momentum,
        gp_rank=args.gp_rank,
        gp_window=args.gp_window,
        gp_update_interval=args.gp_update_interval,
        gp_weight_temperature=args.gp_weight_temperature,
        curvature_floor=args.curvature_floor,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    train_recoder = MomentHubs()
    val_recoder = MomentHubs()
    current_batch_size = args.batch_size

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            disable_running_stats(model)
            perturbed_loss = smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing)
            perturbed_loss.mean().backward()
            optimizer.second_step(zero_grad=True)
            optimizer.update_surrogate(perturbed_loss.mean().item())

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                lr_value = optimizer.param_groups[0]["lr"]
                log(model, loss.cpu(), correct.cpu(), lr_value)
            train_recoder.append("loss", loss.mean().cpu().item())
            train_recoder.append("eta_scale", optimizer.last_eta_scale)
            train_recoder.append("sigma2", optimizer.last_sigma2)

        model.eval()
        log.eval(len_dataset=len(dataset.test), need_flush=True)
        with torch.no_grad():
            _correct = 0
            total = 0
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
                total += targets.size(0)
                _correct += torch.sum(correct).detach().cpu().item()
                val_recoder.append("loss", loss.mean().cpu().item())

            acc = 100.0 * _correct / total
            if acc > best_acc:
                print("Saving..")
                state = {"net": model.state_dict(), "acc": acc, "epoch": epoch}
                torch.save(state, f"./checkpoint/{args.arch}/{args.dataset}/{args.exp_name}.pth")
                best_acc = acc

        proposed_batch_size = int(optimizer.suggested_batch_size)
        if proposed_batch_size != current_batch_size:
            current_batch_size = proposed_batch_size
            rebuild_train_loader(dataset, current_batch_size, args.threads)
            print(f"[CURV-ADAPT] Updated train batch size to {current_batch_size}")

        train_recoder.step_summary()
        val_recoder.step_summary()

    train_recoder.to_csv(f"train_curv_adapt_{args.arch}_{args.dataset}_{args.exp_name}.csv")
    val_recoder.to_csv(f"test_curv_adapt_{args.arch}_{args.dataset}_{args.exp_name}.csv")
    log.flush()
