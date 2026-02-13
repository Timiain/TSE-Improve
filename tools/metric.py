import numpy as np
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
