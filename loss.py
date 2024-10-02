'''
-*- coding:utf-8 -*-
@File:loss.py
'''
import torch
from torch import nn

def loss_fun(output, target):
    output = output.permute(0, 2, 3, 1)
    target = target.permute(0, 2, 3, 1)

    loss_fun_confidence = nn.L1Loss()
    loss_confidence = loss_fun_confidence(output[..., 0], target[..., 0])
    loss = loss_confidence
    return loss