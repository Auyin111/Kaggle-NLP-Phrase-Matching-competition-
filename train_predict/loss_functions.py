import torch
from torch import nn

# ====================================================
# Pearson correlation coefficient loss
# ====================================================
class PCCLoss(nn.Module):
    def __init__(self):
        super(PCCLoss, self).__init__()
        self.sum = torch.sum
        self.pow = torch.pow
        self.mean = torch.mean
        self.var = torch.var
        self.std = torch.std

    def forward(self, pred, label):
        pred = torch.squeeze(pred).t().type(torch.float64)
        label = torch.squeeze(label).t().type(torch.float64)
        pred_mean = self.mean(pred)
        label_mean = self.mean(label)
        pred_var = self.var(pred, unbiased=True)
        label_var = self.var(label, unbiased=True)
        pred_std = self.std(pred, unbiased=True)
        label_std = self.std(label, unbiased=True)
        cov = self.sum((pred - pred_mean) * (label - label_mean)) / (len(pred) - 1)
        pcc = cov / (pred_std * label_std)
        return (1 - pcc).cuda() if torch.cuda.is_available() else (1 - pcc)


# ====================================================
# Concordance Correlation Coefficient loss ver 1
# ====================================================
class CCCLoss1(nn.Module):
    def __init__(self):
        super(CCCLoss1, self).__init__()
        self.sum = torch.sum
        self.mean = torch.mean
        self.var = torch.var
        self.std = torch.std

    def forward(self, pred, label):
        pred = torch.squeeze(pred).t().type(torch.float64)
        label = torch.squeeze(label).t().type(torch.float64)
        pred_mean = self.mean(pred)
        label_mean = self.mean(label)
        pred_var = self.var(pred, unbiased=True)
        label_var = self.var(label, unbiased=True)
        pred_std = self.std(pred, unbiased=True)
        label_std = self.std(label, unbiased=True)
        cov = self.sum((pred - pred_mean) * (label - label_mean)) / (len(pred) - 1)
        pcc = cov / (pred_std * label_std)
        ccc = (2 * cov) / (pred_var + label_var + (pred_mean - label_mean) ** 2 + 1e-6)
        # print(label_var)
        # print(pred_var)
        # print(label_std)
        # print(pred_std)
        # print(cov)
        # print(pcc)
        return (1 - ccc).cuda() if torch.cuda.is_available() else (1 - ccc)


# ====================================================
# Concordance Correlation Coefficient loss ver 2 (Larger range)
# ====================================================
class CCCLoss2(nn.Module):
    def __init__(self):
        super(CCCLoss2, self).__init__()
        self.sum = torch.sum
        self.mean = torch.mean
        self.var = torch.var
        self.std = torch.std

    def forward(self, pred, label):
        pred = torch.squeeze(pred).t().type(torch.float64)
        label = torch.squeeze(label).t().type(torch.float64)
        pred_mean = self.mean(pred)
        label_mean = self.mean(label)
        pred_var = self.var(pred, unbiased=True)
        label_var = self.var(label, unbiased=True)
        pred_std = self.std(pred, unbiased=True)
        label_std = self.std(label, unbiased=True)
        cov = self.sum((pred - pred_mean) * (label - label_mean))
        ccc = (2 * cov) / (pred_var + label_var + (pred_mean - label_mean) ** 2 + 1e-6)

        return (-ccc).cuda() if torch.cuda.is_available() else (-ccc)