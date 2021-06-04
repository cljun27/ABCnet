import numpy as np
import os
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch
import torch.nn.init
from torch.autograd import Variable


class smooothing_loss(nn.Module):
    def __init__(self):
        super(smooothing_loss, self).__init__()

    def forward(y_pred):
        dy = torch.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
        dx = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dz = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])

        dx = torch.mul(dx, dx)
        dy = torch.mul(dy, dy)
        dz = torch.mul(dz, dz)

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d/2.0

class SegLoss_pathch(torch.nn.Module):
    def __init__(self):
        super(SegLoss_pathch,self).__init__()

    def forward(self,a, b, ts, mina, maxa, minb, maxb):
        nb_sample=5000
        nb_patch=0
        cor_sum=0
        a = a*(maxa - mina) + mina
        b = b*(maxb - minb) + minb
        c = a/b
        c_t=c*ts
        c_t=torch.squeeze(c_t)
        list_patch=random.sample(list(torch.nonzero(c_t)), nb_sample)
        for nb_patch in range(nb_sample):
            indice_sample=list_patch[nb_patch]
            [x_pos, y_pos, z_pos]= indice_sample

            c_s= c_t[x_pos-1:x_pos+2,y_pos-1:y_pos+2,z_pos-1:z_pos+2]
            c_flat=torch.flatten(c_s)
            c_nz=c_flat[c_flat.nonzero()]
            c_var = torch.var(c_nz)
            c_mean = torch.mean(c_nz)
            cor = c_var/c_mean
            cor_sum=cor_sum+cor
            nb_patch=nb_patch+1
        
        return cor_sum/nb_sample

class SegLoss_total(torch.nn.Module):
    def __init__(self):
        super(SegLoss_total,self).__init__()

    def forward(self,a, b, ts, mina, maxa, minb, maxb):
       
        a = a*(maxa - mina) + mina

        b = b*(maxb - minb) + minb

        c = a/b
        c_t = torch.masked_select(c, ts)

        c_var = torch.var(c_t)
        c_mean = torch.mean(c_t)
        cor = c_var/c_mean

        return cor

class SegLoss(torch.nn.Module):
    def __init__(self):
        super(SegLoss,self).__init__()

    def forward(self,a, b, ts, mina, maxa, minb, maxb):
       
        a = a*(maxa - mina) + mina

        b = b*(maxb - minb) + minb

        c = a/b
        c_t = torch.masked_select(c, ts)

        c_var = torch.var(c_t)
        c_mean = torch.mean(c_t)
        cor = c_var/c_mean

        return cor

class CorLoss(torch.nn.Module):
    def __init__(self):
        super(CorLoss,self).__init__()

    def forward(self,input, target):
        sigma=1.4
        ret = 1 - torch.exp(-((input - target) ** 2) / (2*(sigma)**2))
        reduction='mean'
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
        return ret
