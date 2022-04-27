import torch.nn as nn
import torchvision.models as models
import torch
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
    
    
class ConvBn(nn.Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c)
        )
        
    def forward(self, x):
        return self.net(x)
    
class ConvBnPrelu(nn.Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBn(in_c, out_c, kernel, stride, padding, groups),
            nn.PReLU(out_c)
        )

    def forward(self, x):
        return self.net(x)
    
class DepthWise(nn.Module):

    def __init__(self, in_c, out_c, kernel=(3, 3), stride=2, padding=1, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnPrelu(in_c, groups, kernel=(1, 1), stride=1, padding=0),
            ConvBnPrelu(groups, groups, kernel=kernel, stride=stride, padding=padding,groups=groups),
            ConvBn(groups, out_c, kernel=(1, 1), stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x)
    
class DepthWiseRes(nn.Module):
    """DepthWise with Residual"""

    def __init__(self, in_c, out_c, kernel=(3, 3), stride=2, padding=1, groups=1):
        super().__init__()
        self.net = DepthWise(in_c, out_c, kernel, stride, padding, groups)

    def forward(self, x):
        return self.net(x) + x   
    
class MultiDepthWiseRes(nn.Module):

    def __init__(self, num_block, channels, kernel=(3, 3), stride=1, padding=1, groups=1):
        super().__init__()

        self.net = nn.Sequential(*[
            DepthWiseRes(channels, channels, kernel, stride, padding, groups)
            for _ in range(num_block)
        ])

    def forward(self, x):
        return self.net(x)    
    
class Net(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()
        self.conv1 = ConvBnPrelu(3, 64, kernel=(3, 3), stride=2, padding=1)
        self.conv2 = ConvBn(64, 64, kernel=(3, 3), stride=1, padding=1, groups=64)
        self.conv3 = DepthWise(64, 64, kernel=(3, 3), stride=2, padding=1, groups=128)
        self.conv4 = MultiDepthWiseRes(num_block=4, channels=64, kernel=3, stride=1, padding=1, groups=128)
        self.conv5 = DepthWise(64, 128, kernel=(3, 3), stride=2, padding=1, groups=256)
        self.conv6 = MultiDepthWiseRes(num_block=6, channels=128, kernel=(3, 3), stride=1, padding=1, groups=256)
        self.conv7 = DepthWise(128, 128, kernel=(3, 3), stride=2, padding=1, groups=512)
        self.conv8 = MultiDepthWiseRes(num_block=2, channels=128, kernel=(3, 3), stride=1, padding=1, groups=256)
        self.conv9 = ConvBnPrelu(128, 512, kernel=(1, 1))
        self.conv10 = ConvBn(512, embedding_size, groups=embedding_size, kernel=(7, 7))
        
        self.flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        # out = self.avgpool(out)
        out = self.flatten(out)
        out=self.linear(out)
        out = self.bn(out)
        return out

# # net=Net(512)
# # y=net(torch.randn(1,1,128,128))
# def l2_norm(input,axis=1):
#     norm = torch.norm(input,2,axis,True)
#     output = torch.div(input, norm)
#     return output    

# class Arcface(nn.Module):
#     # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
#     def __init__(self, embedding_size, classnum,  s=30., m=0.5):
#         super(Arcface, self).__init__()
#         self.classnum = classnum
#         self.kernel = nn.Parameter(torch.Tensor(embedding_size,classnum))
#         # initial kernel
#         self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
#         self.m = m # the margin value, default is 0.5
#         self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.mm = self.sin_m * m  # issue 1
#         self.threshold = math.cos(math.pi - m)
#     def forward(self, embbedings, label):
#         # weights norm
#         nB = len(embbedings)
#         kernel_norm = l2_norm(self.kernel,axis=0)
#         # cos(theta+m)
#         cos_theta = torch.mm(embbedings,kernel_norm)
# #         output = torch.mm(embbedings,kernel_norm)
#         cos_theta = cos_theta.clamp(-1,1) # for numerical stability
#         cos_theta_2 = torch.pow(cos_theta, 2)
#         sin_theta_2 = 1 - cos_theta_2
#         sin_theta = torch.sqrt(sin_theta_2+1e-8)
#         cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
#         # this condition controls the theta+m should in range [0, pi]
#         #      0<=theta+m<=pi
#         #     -m<=theta<=pi-m
#         cond_v = cos_theta - self.threshold
#         cond_mask = cond_v <= 0
#         keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
#         cos_theta_m[cond_mask] = keep_val[cond_mask]
#         output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
#         idx_ = torch.arange(0, nB, dtype=torch.long)
#         output[idx_, label] = cos_theta_m[idx_, label]
#         output *= self.s # scale up in order to make softmax work, first introduced in normface
#         return output
class Arcface(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # fix nan problem:
        sine = torch.sqrt(torch.clamp((1.0 - torch.pow(cosine, 2)),1e-9,1))

        phi = cosine * self.cos_m - sine * self.sin_m
        
        # print(cosine)
        # phi=torch.cos(torch.acos(cosine)+self.m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output    
    
    
    

