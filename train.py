import os
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
import torch.nn.functional as F
from model import Net,Arcface
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dataset import MYDataset
from config import Config
import math
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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


net=Net(512).cuda()
arc_margin=Arcface(512,len(Config.dirs)).cuda()


# t_batch_size=60
train_loader=torch.utils.data.DataLoader(my_data,shuffle=True,batch_size=Config.train_batch_size,num_workers=16,pin_memory=True)

focalloss=FocalLoss()

optimizer =torch.optim.SGD([{'params': net.parameters()}, {'params': arc_margin.parameters()}], 
                            lr=Config.lr, weight_decay=Config.weight_decay)

# scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)	# base_lr = 0.1

warm_up_iter = 10
T_max = 100	
lr_max = 0.1	
lr_min = 1e-5

lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
        (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)

best_result=[0,0]
num_epochs=100
for epoch in range(num_epochs):
    

    
    num_step=len(train_loader)
    for step, (x, y) in enumerate(train_loader):

            train_loss=[]
            train_acc=[]
            x = x.cuda()
            y = y.cuda()
            ys = net(x) 
            
            m=arc_margin(ys,y)
            loss= focalloss(m, y)
            
            iter_loss = loss.item()

       
               
            ############ BACK PROP #################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss)
            
            if step%1000==0:
                value =  torch.argmax(m, dim=1)
                acc =  torch.sum((value == y).float()) / len(y)
                print('acc :{}'.format(acc))
                
                print ('Epoch {}/{}, Step {}/{},  Training Loss: {:.3f}'
                            .format(epoch+1,num_epochs, step,num_step, sum(train_loss)/len(train_loss)) )  
        
    scheduler.step()

    torch.save(net.state_dict(), '{}best.pt'.format(epoch))
