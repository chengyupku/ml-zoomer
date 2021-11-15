from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch
num_classes = 10
scores = torch.tensor([0.1, 0.2, 0.8, 0.95, 1.05, 0.05, 0.5, 0.4, 0.3, 0.2, 0.35, 0.86])
labels = torch.tensor([0, 0, 1, 1, 1, 0, 1, 1 ,0, 1, 1, 0])
# ones = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# zeros = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
ones = torch.ones(12, dtype=torch.int32)
zeros = torch.zeros(12, dtype=torch.int32)
def auc_calculate(scores, labels, bucket_num=20):
    gaps = 1/bucket_num
    lastFPR = 0
    area = 0
    for i in range(bucket_num, 0, -1):
        # i = 
        threshold = i*gaps
        TP = (torch.where(scores>=threshold, ones, zeros) * torch.where(labels==1, ones, zeros)).sum()
        FP = (torch.where(scores>=threshold, ones, zeros) * torch.where(labels==0, ones, zeros)).sum()
        TN = (torch.where(scores<threshold, ones, zeros) * torch.where(labels==0, ones, zeros)).sum()
        FN = (torch.where(scores<threshold, ones, zeros) * torch.where(labels==1, ones, zeros)).sum()
        # print(TP)
        # print(FP)
        # print(TN)
        # print(FN)
        TPR = TP / (TP+FN)
        FPR = FP / (FP+TN)
        print("TPR", TPR)
        print("FPR", FPR)
        area += TPR*(FPR-lastFPR)
        lastFPR = FPR
    return area

print(auc_calculate(scores, labels))