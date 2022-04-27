import os
import torch
import torch.nn as nn
from model import Net,ArcMarginProduct
from dataset import MYDataset
from config import Config
import math
import datetime
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

from models import resnet18

net=resnet18().cuda()

arc_margin=ArcMarginProduct(512,len(Config.dirs),s=Config.arc_s,m=Config.arc_m).cuda()

t_batch_size=Config.train_batch_size

train_loader=torch.utils.data.DataLoader(my_data,shuffle=True,batch_size=t_batch_size,num_workers=16,pin_memory=True)

focalloss=FocalLoss()

optimizer =torch.optim.SGD([{'params': net.parameters()}, {'params': arc_margin.parameters()}], 
                            lr=Config.base_lr, weight_decay=Config.weight_decay)

scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)




# warm_up_iter = int(Config.num_epochs*0.1)
# T_max = Config.num_epochs
# lr_max = Config.base_lr
# lr_min = 1e-5

# lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
#         (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1

# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)



net = nn.DataParallel(net)
arc_margin=nn.DataParallel(arc_margin)

if Config.load_net==True:
    net.load_state_dict(torch.load('/home/xu/XU/my_rec/save_emb/119best.pt'))
    arc_margin.load_state_dict(torch.load('/home/xu/XU/my_rec/save_arc/119best.pt'))

from check import lfw_test
for epoch in range(Config.num_epochs):
    
    
    
    num_step=len(train_loader)
    for step, (x, y) in enumerate(train_loader):

            train_loss=[]
            train_acc=[]
            x = x.cuda()
            y = y.cuda()
            ys = net(x) 
            
            m=arc_margin(ys,y)
            loss= focalloss(m, y)
            loss=loss.mean()
            
            iter_loss = loss.item()

            # value =  torch.argmax(m, dim=1)
            # acc =  torch.sum((value == y).float()) / len(y)
               
            ############ BACK PROP #################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss)
            # train_acc.append(acc)
            if step%Config.num_iter_show==0:
    
                value =  torch.argmax(m, dim=1)
                acc =  torch.sum((value == y).float()) / len(y)
                print(datetime.datetime.now())
                print('acc:{}'.format(acc))
                print ('Epoch {}/{}, Step {}/{},  Training Loss: {:.3f}'
                            .format(epoch+1,Config.num_epochs, step,num_step, sum(train_loss)/len(train_loss)) )  
                lfw_test(net)
        
    scheduler.step()

    torch.save(net.state_dict(), './save_emb/{}best.pt'.format(epoch))
    torch.save(arc_margin.state_dict(), './save_arc/{}best.pt'.format(epoch))

