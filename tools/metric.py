import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ListRecorder(object):
    """A generate recodedr tool"""
    def __init__(self):
        self.data={}
    
    def append(self,key,value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def to_csv(self,name):
        #补全大小
        max =0
        for key in self.data:
            if len(self.data[key])>max:
                max = len(self.data[key])
        
        for key in self.data:
            diff =max -  len(self.data[key])
            for i in range(diff):
                self.data[key].append(-1)

        import pandas as pd
        df = pd.DataFrame.from_dict(self.data)
        df.to_csv(name)    

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def OA(self):
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    def Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp)
        return IoU

    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

    def Pixel_Accuracy_Class(self):
        #         TP                                  TP+FP
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return Acc

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

class MomentHubs(object):
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.meters={}
        self.recoder=ListRecorder()
    
    def get(self, key):
        if key not in self.meters:
            item = MomentMeter()
            self.meters[key]=item
            return self.meters[key]
        else:
            return self.meters[key]
    
    def append(self,key,value):
        item = self.get(key)   
        item.append(value) 
    
    def append_direct_to_recoder(self,key,value):
        self.recoder.append(key,value)
    
    def reset_all_meters(self):
        for key in self.meters:
            self.get(key).reset()

    def step_summary(self):
        for key in self.meters:
            item=self.get(key)
            item.summary()
            self.recoder.append("{}_{}".format(key,'mean'), item.mean)
            self.recoder.append("{}_{}".format(key,'variance'), item.variance)
            self.recoder.append("{}_{}".format(key,'skewness'), item.skewness)
            self.recoder.append("{}_{}".format(key,'kurtosis'), item.kurtosis)
        
        self.reset_all_meters()
        self.meters={}

    def to_csv(self, name):
        self.recoder.to_csv(name)
    
class MomentMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.mean = 0
        self.variance = 0
        self.skewness = 0
        self.kurtosis = 0
        self.items=[]

    def append(self, val, n=1):
        if isinstance(val, list):
            self.items.extend(val)
        else:
            self.items.append(val)
    
    def summary(self):
        data = np.array(self.items)
        mean = np.mean(data, axis=0)
        variance = np.var(data, axis=0)
        skewness = np.mean(((data - mean)**3) / (np.sqrt(variance)**3), axis=0)
        kurtosis = np.mean(((data - mean)**4) / (variance**2), axis=0)
        self.mean=mean
        self.variance = variance
        self.skewness = skewness
        self.kurtosis = kurtosis

class SegCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1, **kwargs):
        super(SegCrossEntropyLoss, self).__init__()
        self.task_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, inputs, targets):
        B, H, W = targets.size()
        inputs = F.interpolate(inputs, (H, W), mode='bilinear', align_corners=True)
        return self.task_loss(inputs, targets)


class GrokkingIndexMeter(object):
    """
    GI (Grokking-Index) estimator:
      GI = min(S_loc, S_int, S_glob) + mean(S_loc, S_int, S_glob)

    组件定义：
      - S_loc: 局部扰动敏感度（权重扰动后训练损失增量）
      - S_int: 插值路径壁垒（插值模型上的 val-train 最大差）
      - S_glob: 模型池最坏训练损失
    """

    def __init__(self, max_models=6, alpha_steps=11):
        self.max_models = max_models
        self.alpha_steps = alpha_steps
        self.model_pool = []

    def _clone_state_dict(self, model):
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def update_model_pool(self, model, train_loss=None):
        self.model_pool.append(
            {
                "state_dict": self._clone_state_dict(model),
                "train_loss": None if train_loss is None else float(train_loss),
            }
        )
        if len(self.model_pool) > self.max_models:
            self.model_pool.pop(0)

    @staticmethod
    def _iterate_batches(data_loader, max_batches):
        for idx, batch in enumerate(data_loader):
            if idx >= max_batches:
                break
            if len(batch) >= 2:
                yield batch[0], batch[1]

    def _average_loss(self, model, data_loader, criterion, device, max_batches=1):
        model.eval()
        losses = []
        with torch.no_grad():
            for inputs, targets in self._iterate_batches(data_loader, max_batches):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                losses.append(float(criterion(outputs, targets).mean().item()))
        return float(np.mean(losses)) if losses else 0.0

    def _local_sensitivity(self, model, train_loader, criterion, device, lambda_loc=0.05, max_batches=1):
        model.train()
        losses = []
        for inputs, targets in self._iterate_batches(train_loader, max_batches):
            inputs = inputs.to(device)
            targets = targets.to(device)

            model.zero_grad(set_to_none=True)
            base_loss = criterion(model(inputs), targets).mean()
            base_loss.backward()

            grad_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm_sq += float(torch.sum(p.grad.detach() ** 2).item())
            grad_norm = np.sqrt(grad_norm_sq) + 1e-12
            rho = lambda_loc * grad_norm

            perturbations = []
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is None:
                        perturbations.append(None)
                        continue
                    eps = p.grad / (p.grad.norm(p=2) + 1e-12) * rho
                    p.add_(eps)
                    perturbations.append(eps)

            with torch.no_grad():
                perturbed_loss = criterion(model(inputs), targets).mean()
            losses.append(float((perturbed_loss - base_loss.detach()).item()))

            with torch.no_grad():
                for p, eps in zip(model.parameters(), perturbations):
                    if eps is not None:
                        p.sub_(eps)

            model.zero_grad(set_to_none=True)

        return float(np.mean(losses)) if losses else 0.0

    def _interpolation_barrier(self, current_model, model_builder, train_loader, val_loader, criterion, device, max_batches=1):
        if not self.model_pool:
            return 0.0

        other_state = self.model_pool[np.random.randint(len(self.model_pool))]["state_dict"]
        current_state = self._clone_state_dict(current_model)

        interp_model = model_builder()
        interp_model.to(device)
        interp_model.eval()

        barrier = -float("inf")
        alphas = np.linspace(0.0, 1.0, num=self.alpha_steps)
        for alpha in alphas:
            mixed_state = {}
            for name in current_state:
                w_cur = current_state[name].to(device)
                w_old = other_state[name].to(device)
                mixed_state[name] = ((1.0 - alpha) * w_cur + alpha * w_old).to(w_cur.dtype)

            interp_model.load_state_dict(mixed_state, strict=True)
            train_loss = self._average_loss(interp_model, train_loader, criterion, device, max_batches=max_batches)
            val_loss = self._average_loss(interp_model, val_loader, criterion, device, max_batches=max_batches)
            barrier = max(barrier, val_loss - train_loss)

        return float(max(barrier, 0.0))

    def _global_worst_case(self, fallback_train_loss):
        train_losses = [item["train_loss"] for item in self.model_pool if item["train_loss"] is not None]
        if fallback_train_loss is not None:
            train_losses.append(float(fallback_train_loss))
        return float(max(train_losses)) if train_losses else 0.0

    def compute(self, model, model_builder, train_loader, val_loader, criterion, device, lambda_loc=0.05, max_batches=1, fallback_train_loss=None):
        s_loc = self._local_sensitivity(model, train_loader, criterion, device, lambda_loc=lambda_loc, max_batches=max_batches)
        s_int = self._interpolation_barrier(model, model_builder, train_loader, val_loader, criterion, device, max_batches=max_batches)
        s_glob = self._global_worst_case(fallback_train_loss)
        components = [s_loc, s_int, s_glob]
        gi = float(np.min(components) + np.mean(components))
        return {
            "gi": gi,
            "s_loc": float(s_loc),
            "s_int": float(s_int),
            "s_glob": float(s_glob),
        }


def gi_decision(gi, dataset_thr=0.8, model_thr=0.7, hyper_thr=0.6):
    if gi > dataset_thr:
        return {
            "level": "dataset",
            "action": "扩充/增强数据，或重采样修正分布偏差",
        }
    if gi > model_thr:
        return {
            "level": "model",
            "action": "降低模型容量并增加正则（Dropout/LayerNorm）",
        }
    if gi > hyper_thr:
        return {
            "level": "hyper",
            "action": "增大对抗强度并使用更平滑学习率调度",
        }
    return {
        "level": "stable",
        "action": "保持当前配置，继续训练或结束迭代",
    }

if __name__ == '__main__':

    gt = np.array([[0, 2, 1],
                   [1, 2, 1],
                   [1, 0, 1]])

    pre = np.array([[0, 1, 1],
                   [2, 0, 1],
                   [1, 1, 1]])

    eval = Evaluator(num_class=3)
    eval.add_batch(gt, pre)
    print(eval.confusion_matrix)
    print(eval.get_tp_fp_tn_fn())
    print(eval.Precision())
    print(eval.Recall())
    print(eval.Intersection_over_Union())
    print(eval.OA())
    print(eval.F1())
    print(eval.Frequency_Weighted_Intersection_over_Union())
