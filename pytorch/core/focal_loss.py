import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from IPython import embed

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-5
        self.size_average = size_average

    def forward(self, predict, target):
        method = nn.Sigmoid()
        predict = method(predict)
        # clamp to prevent log(0) => nan
        predict = predict.clamp(self.eps, 1 - self.eps) 

        loss=-target*(1-self.alpha)*((1-predict)*self.gamma)*torch.log(predict)-(1-target)*self.alpha*(predict**self.gamma)*torch.log(1-predict)
        if self.size_average: return loss.mean()
        else: return loss.sum()
