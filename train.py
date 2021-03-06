import os
import torch
import torch.nn as nn
from model import Net,Arcface
from dataset import MYDataset
from config import Config
import math
import datetime
from check import lfw_test

from find_lr import find_lr
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

if Config.load:
    net.load_state_dict(torch.load( './save_emb/13best.pt'))
    arc_margin.load_state_dict(torch.load('./save_arc/13best.pt'))

# warm_up_iter = int(Config.num_epochs*0.1)
# T_max = Config.num_epochs
# lr_max = Config.base_lr	
# lr_min = 1e-4

# lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
#         (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1


# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)

scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Config.num_epochs)
# import matplotlib.pyplot as plt


start_epoch=0

if  Config.load:
    start_epoch=13

best_acc=0
best_epoch=0
for epoch in range(start_epoch,Config.num_epochs):
    
    num_step=len(train_loader)
    for step, (x, y) in enumerate(train_loader):
            net.train()

            train_loss=[]
            train_acc=[]
            x = x.cuda()
            y = y.cuda()
            ys = net(x) 
            
            m=arc_margin(ys,y)
            loss= nn.CrossEntropyLoss()(m, y)
            if Config.multi_gpu==True:
                loss=loss.mean()
            
            iter_loss = loss.item()

            ############ BACK PROP #################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss)
            
            if step%Config.num_iter_show==0:
    
                value =  torch.argmax(m, dim=1)
                acc =  torch.sum((value == y).float()) / len(y)
                print(datetime.datetime.now())
                print('acc:{}'.format(acc))
                print ('Epoch {}/{}, Step {}/{},  Training Loss: {:.3f}'
                            .format(epoch+1,Config.num_epochs, step,num_step, sum(train_loss)/len(train_loss)) )  
                acc=lfw_test(net)
                if acc>best_acc:
                    best_acc=acc
                    best_epoch=epoch
                print('best acc :{}, best epoch :{}'.format(best_acc,best_epoch))

    scheduler.step()

    torch.save(net.state_dict(), './save_emb2/{}best.pt'.format(epoch))
    torch.save(arc_margin.state_dict(), './save_arc2/{}best.pt'.format(epoch))
