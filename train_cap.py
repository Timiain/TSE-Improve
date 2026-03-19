import argparse
import os

import torch

from data.cifar import Cifar, Cifar100
from model import ResNet_cifar
from model import vit
from model.PyramidNet import PyramidNet
from model.smooth_cross_entropy import smooth_crossentropy
from model.wide_res_net import WideResNet
from sam import CAP
from tools.metric import MomentHubs
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="Use ASAM-style parameter scaling.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in training and validation.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--cuda", default=0, type=int, help="GPU index.")
    parser.add_argument("--exp_name", default="cap", type=str, help="Experiment name.")
    parser.add_argument("--arch", default="wrn28_10", type=str, help="Model name.")
    parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset.")
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint name.")
    parser.add_argument("--num_classes", default=10, type=int, help="Number of classes.")

    parser.add_argument("--c", default=0.1, type=float, help="Curvature-to-radius scaling constant.")
    parser.add_argument("--alpha_min", default=1e-3, type=float, help="Minimum adaptive perturbation radius.")
    parser.add_argument("--alpha_max", default=0.2, type=float, help="Maximum adaptive perturbation radius.")
    parser.add_argument("--curvature_ema", default=0.05, type=float, help="EMA factor for tau/lambda smoothing.")
    parser.add_argument("--lr_coupling", default=1.0, type=float, help="Beta term for lr = lr0 / (1 + beta * alpha).")
    parser.add_argument("--power_iter_steps", default=1, type=int, help="Power-iteration steps for dominant eigenvalue estimate.")
    parser.add_argument("--stability_lipschitz", default=10.0, type=float, help="Lipschitz proxy used to enforce lr * alpha stability.")
    parser.add_argument("--k_sync", default=100, type=int, help="Hyperparameter synchronization interval.")
    parser.add_argument("--gamma_decay", default=0.95, type=float, help="Decay applied when smoothing factor is rescaled.")
    parser.add_argument("--c_decay", default=0.95, type=float, help="Decay applied when c is rescaled.")
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

    optimizer = CAP(
        model.parameters(),
        torch.optim.SGD,
        adaptive=args.adaptive,
        c=args.c,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        curvature_ema=args.curvature_ema,
        lr_coupling=args.lr_coupling,
        power_iter_steps=args.power_iter_steps,
        stability_lipschitz=args.stability_lipschitz,
        k_sync=args.k_sync,
        gamma_decay=args.gamma_decay,
        c_decay=args.c_decay,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    train_recoder = MomentHubs()
    val_recoder = MomentHubs()

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            def closure():
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing).mean()
                return loss

            clean_loss, perturbed_loss = optimizer.step(closure)

            with torch.no_grad():
                predictions = model(inputs)
                correct = torch.argmax(predictions.data, 1) == targets

            with torch.no_grad():
                log(model, clean_loss.cpu(), correct.cpu(), optimizer.last_lr)
            train_recoder.append("loss", float(clean_loss.detach().cpu().item()))
            train_recoder.append("perturbed_loss", float(perturbed_loss.cpu().item()))
            train_recoder.append("alpha", optimizer.last_alpha)
            train_recoder.append("tau", optimizer.last_tau)
            train_recoder.append("lambda", optimizer.last_lambda)
            train_recoder.append("lr", optimizer.last_lr)

        model.eval()
        log.eval(len_dataset=len(dataset.test), need_flush=True)

        with torch.no_grad():
            total = 0
            correct_total = 0
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
                total += targets.size(0)
                correct_total += torch.sum(correct).detach().cpu().item()
                val_recoder.append("loss", loss.mean().cpu().item())

            acc = 100.0 * correct_total / total
            if acc > best_acc:
                print("Saving..")
                state = {
                    "net": model.state_dict(),
                    "acc": acc,
                    "epoch": epoch,
                }
                torch.save(state, f"./checkpoint/{args.arch}/{args.dataset}/{args.exp_name}.pth")
                best_acc = acc

        train_recoder.step_summary()
        val_recoder.step_summary()

    train_recoder.to_csv(f"train_cap_{args.arch}_{args.dataset}_{args.exp_name}.csv")
    val_recoder.to_csv(f"test_cap_{args.arch}_{args.dataset}_{args.exp_name}.csv")
    log.flush()
