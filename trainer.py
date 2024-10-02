'''
-*- coding:utf-8 -*-
@File:trainer.py
'''
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import *
from network import *
from loss import *

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

dataset = SpinEchoDataSet()
data_loader = DataLoader(dataset, batch_size=50, shuffle=True)

net = SpinEchoNet().to(device)

opt = optim.Adam(net.parameters(), lr=0.01)

lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3, eps=0.0005)

def train():
    # model training
    global loss
    epoch = 30
    for i in range(epoch):
        for index_batch, (data, target) in enumerate(data_loader):

            data, target = data.to(device), target.to(device)

            output, _ = net(data)

            loss = loss_fun(output.float(), target.float())

            opt.zero_grad()

            loss.backward()

            opt.step()

            print(f'epoch:{i}  batch:{index_batch}  loss:', loss.item())

        # save model
        torch.save(net.state_dict(), f'params/net_{i + 1}.pt')
        print('model saves successfully')

        lrScheduler.step(loss)
        lr = opt.param_groups[0]['lr']
        print("epoch={}, lr={}".format(epoch, lr))
    print('Finished Training')

if __name__ == "__main__":
    train()
