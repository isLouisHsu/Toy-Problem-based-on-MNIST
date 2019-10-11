# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-11 10:09:56
@LastEditTime: 2019-10-11 12:03:39
@Update: 
'''
import os
import numpy as np
from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch import optim
import torch.cuda as cuda
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from datasets import MNIST
from models import GeneratorNet, DiscriminatorNet
from processbar import ProcessBar

def train(batchsize=128, feature_size=32, lr_g=4e-5, lr_d=1e-3, n_epoches=100, milestones=[40, 80]):

    ## 数据
    mnistdata = MNIST()
    mnistloader = DataLoader(mnistdata, batchsize, True, drop_last=True)

    ## 网络
    GNet = GeneratorNet(feature_size)
    DNet = DiscriminatorNet()
    if cuda.is_available():
        GNet.cuda()
        DNet.cuda()
    
    ## 损失
    criterion = nn.BCELoss()
    
    ## 优化器
    optimizerG = optim.SGD(GNet.parameters(), lr_g)
    optimizerD = optim.SGD(DNet.parameters(), lr_d)
    schedulerG = MultiStepLR(optimizerG, milestones)
    schedulerD = MultiStepLR(optimizerD, milestones)

    ## 日志
    writer = SummaryWriter('../log/GAN_MNIST')
    bar    = ProcessBar(n_epoches)

    for i_epoch in range(n_epoches):

        bar.step()
        schedulerG.step()
        schedulerD.step()

        lossG = []; lossD = []

        for i_batch, (realImg, _) in enumerate(mnistloader):

            ## 生成对应标签
            Ones  = torch.ones (batchsize).float()
            Zeros = torch.zeros(batchsize).float()

            ## 生成虚假图片
            noise = torch.randn(batchsize, feature_size)

            if cuda.is_available():
                noise    = noise.cuda()
                realImg  = realImg.cuda()
                Ones     = Ones.cuda()
                Zeros    = Zeros.cuda()

            fakeImg = GNet(noise)

            ## 计算真实图片的鉴别损失，希望其为`1`
            pred_real = DNet(realImg)
            lossD_real = criterion(pred_real, Ones )

            ## 计算虚假图片的鉴别损失，希望其为`0`
            pred_fake = DNet(fakeImg)
            lossD_fake = criterion(pred_fake, Zeros)

            ## 计算鉴别器损失，更新鉴别器参数
            lossD_i = (lossD_real + lossD_fake) / 2
            optimizerD.zero_grad()
            lossD_i.backward()
            optimizerD.step()

            ## 重新生成图片
            noise = torch.randn(batchsize, feature_size)
            if cuda.is_available():
                noise    = noise.cuda()
            fakeImg = GNet(noise)
            
            ## 计算生成器损失，希望鉴别器得到`1`
            pred_fake = DNet(fakeImg)
            lossG_i = criterion(pred_fake, Ones )
            optimizerG.zero_grad()
            lossG_i.backward()
            optimizerG.step()
            
            lossG += [lossG_i.detach().cpu().numpy()]
            lossD += [lossD_i.detach().cpu().numpy()]

        ## 日志
        lossG = np.mean(lossG); lossD = np.mean(lossD)
        writer.add_scalars('loss', {'G': lossG, 'D': lossD}, i_epoch)
        writer.add_images('image', fakeImg[:12], i_epoch)

    writer.close()

if __name__ == "__main__":
    train()
