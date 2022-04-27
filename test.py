import math
import torch
from torchvision.models import resnet18

model = resnet18()	# 加载模型
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)	# base_lr = 0.1

warm_up_iter = 10
T_max = 100	
lr_max = 0.1	
lr_min = 1e-5

lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
        (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)

for epoch in range(50):
    print(optimizer.param_groups[0]['lr'])
    optimizer.step()
    scheduler.step()
