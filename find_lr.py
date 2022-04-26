import os
import torch
import torch.nn as nn
from model import Net,Arcface
from dataset import MYDataset
from config import Config
import math
import datetime
from check import lfw_test

if Config.multi_gpu==True:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class FocalLoss(nn.Module):

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
    

my_data=MYDataset()

device=torch.device('cuda')




net=Net(Config.embedding_size).cuda()

arc_margin=Arcface(Config.embedding_size,len(Config.dirs)).cuda()

if Config.multi_gpu==True:
    net = nn.DataParallel(net)
    arc_margin=nn.DataParallel(arc_margin)


train_loader=torch.utils.data.DataLoader(my_data,shuffle=True,batch_size=Config.train_batch_size,num_workers=16,pin_memory=True)

focalloss=FocalLoss()

optimizer =torch.optim.SGD([{'params': net.parameters()}, {'params': arc_margin.parameters()}], 
                            lr=Config.base_lr, weight_decay=Config.weight_decay)



import torch.nn.functional as F
import math
from tqdm import tqdm

def find_lr(init_value = 1e-8, final_value=10., beta = 0.98):
    num = len(train_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in tqdm(train_loader):
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        
    
        ys = net(inputs) 
        
        m=arc_margin(ys,labels)
        loss= focalloss(m, labels)

        
        
        
        
        
        
        
        # outputs = net(inputs)
        # loss = F.nll_loss(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


import matplotlib.pyplot as plt


logs,losses = find_lr()
plt.plot(logs[10:-5],losses[10:-5])












