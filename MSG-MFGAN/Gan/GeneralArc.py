import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from enum import Enum

dim_dict={4:256, 8:256, 16:256, 32:128, 64:128, 128:64, 256:64,512:32}
#dim_dict={4:128, 8:128, 16:128, 32:64, 64:64, 128:32, 256:32,512:16}
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
       
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out


class MiniSiameseBlock(nn.Module):
    """ MiniSiamese Block with Densenet"""
    def __init__(self,in_size,att=False):
        super(MiniSiameseBlock,self).__init__()
        self.in_size=in_size
        dim = dim_dict[in_size]
        self.dim=dim  
        self.lr=nn.LeakyReLU()
       
        self.conv1=nn.Conv2d(in_channels = 3 , out_channels = dim//16 , kernel_size= 3,stride=1,padding=1)
        if att:
            self.att=Self_Attn(dim//16)
        else:
            self.att=nn.Identity()
        self.conv2=nn.Conv2d(in_channels = dim//16 , out_channels = dim//8 , kernel_size= 3,stride=1,padding=1)
        
        self.conv3=nn.Conv2d(in_channels = 3*dim//16 , out_channels = dim//4 , kernel_size= 3,stride=1,padding=1)
        
        
        
    def forward(self,x):
        y=self.conv1(x)
        temp=self.lr(y)        #first conv
        
        out=self.att(temp)     #self-attention
        
       
        out=self.conv2(out)#second conv
        out=self.lr(out)
            
        out=torch.cat([out,temp],dim=1)#concat
            
        out=self.conv3(out)#last conv
        out=self.lr(out)
        
          
        return out

class MaxSiameseBlock(nn.Module):
    """ MaxSiamese Block with Densenet"""
    def __init__(self,in_size,att=False):
        super(MaxSiameseBlock,self).__init__()
        self.in_size=in_size
        dim = dim_dict[in_size]
        self.dim=dim  
        self.lr=nn.LeakyReLU()
        
        self.conv1=nn.Conv2d(in_channels = 3 , out_channels = dim//8 , kernel_size= 3,stride=2,padding=1)
        if att:
            self.att=Self_Attn(dim//8)
        else:
            self.att=nn.Identity()
        
            
        self.conv2=nn.Conv2d(in_channels = dim//8 , out_channels = dim//4 , kernel_size= 3,stride=1,padding=1)
            
        self.up=nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3=nn.Conv2d(in_channels = 3*dim//8 , out_channels = dim//4 , kernel_size= 3,stride=1,padding=1)
         
        
        
    def forward(self,x):
        y=self.conv1(x)
        temp=self.lr(y)        #first conv
        
        out=self.att(temp)     #self-attention
        
        
        
        out=self.conv2(out)#second conv
        out=self.lr(out)
            
        out=torch.cat([out,temp],dim=1)#concat
            
        out=self.up(out)
            
        out=self.conv3(out)#last conv
        out=self.lr(out)
          
        return out

def SiameseBlock(in_size):
    # if in_size <= 32:
    #     return MiniSiameseBlock(in_size)
    # else:
    #     return MaxSiameseBlock(in_size)
    return MiniSiameseBlock(in_size)

class Fusionblock(nn.Module):
    """ Fusion Layer with cSE and max combine  with lower feature""" 
    def __init__(self,in_size,cSE=True):
        super(Fusionblock,self).__init__()
     
        self.in_size=in_size
        self.lrelu=nn.LeakyReLU()
        self.dim=dim_dict[in_size]
        if in_size>=8:
            lower_dim=dim_dict[in_size//2]
        self.siam=SiameseBlock(in_size)
        
        self.conv2=nn.Conv2d(in_channels = lower_dim , out_channels = self.dim//4 , kernel_size= 3,stride=1,padding=1)
        self.conv=nn.Conv2d(in_channels = self.dim , out_channels = self.dim , kernel_size= 3,stride=1,padding=1)
        if cSE:
            self.cSE=ChannelSELayer(self.dim)
        else:
            self.cSE=lambda x:x
        

        
    def forward(self,x,y,lower_feature=None):
        x=self.siam(x)
        y=self.siam(y)
        #max combine the feature
        z=torch.max(x,y)
        
        lower_feature=self.conv2(lower_feature)
        lower_feature=self.lrelu(lower_feature)
        z=torch.cat([z,lower_feature,x,y],dim=1)
        
        z=self.conv(z)
        z=self.cSE(z)
        z=self.lrelu(z)
        return z

class InitFusionblock(nn.Module):
    """ Fusion Layer with cSE and max combine without lower feature""" 
    def __init__(self,in_size,cSE=True):
        super(InitFusionblock,self).__init__()
      
        self.in_size=in_size
        self.dim=dim_dict[in_size]
        self.lrelu=nn.LeakyReLU()
        self.siam=SiameseBlock(in_size)
      
      
        self.conv=nn.Conv2d(in_channels = 3*self.dim//4 , out_channels = self.dim , kernel_size= 3,stride=1,padding=1)
        
        if cSE:
            self.cSE=ChannelSELayer(self.dim)
        else:
            self.cSE=lambda x:x
        

        
    def forward(self,x,y,lower_feature=None):
        x=self.siam(x)
        y=self.siam(y)
        
        
        #max combine the feature
        z=torch.max(x,y)
        
       
        z=torch.cat([z,x,y],dim=1)
        z=self.conv(z)
        z=self.cSE(z)
        z=self.lrelu(z)
        return z        

class NomaxFusionblock(nn.Module):
    """ Fusion Layer with cSE and max combine  with lower feature""" 
    def __init__(self,in_size,cSE=True):
        super(NomaxFusionblock,self).__init__()
     
        self.in_size=in_size
        self.dim=dim_dict[in_size]
        self.lrelu=nn.LeakyReLU()
        if in_size>=8:
            lower_dim=dim_dict[in_size//2]
        self.siam=SiameseBlock(in_size)
        
        self.conv2=nn.Conv2d(in_channels = lower_dim , out_channels = self.dim//4 , kernel_size= 3,stride=1,padding=1)
        self.conv=nn.Conv2d(in_channels = 3*self.dim//4 , out_channels = self.dim , kernel_size= 3,stride=1,padding=1)
    
        if cSE:
            self.cSE=ChannelSELayer(self.dim)
        else:
            self.cSE=lambda x:x
        

        
    def forward(self,x,y,lower_feature=None):
        x=self.siam(x)
        y=self.siam(y)
       
        
        lower_feature=self.conv2(lower_feature)
        lower_feature=self.lrelu(lower_feature)
        z=torch.cat([lower_feature,x,y],dim=1)
        
        z=self.conv(z)
        z=self.cSE(z)
        z=self.lrelu(z)
        return z

class NomaxInitFusionblock(nn.Module):
    """ Fusion Layer with cSE and max combine without lower feature""" 
    def __init__(self,in_size,cSE=True):
        super(NomaxInitFusionblock,self).__init__()
      
        self.in_size=in_size
        self.dim=dim_dict[in_size]
        self.lrelu=nn.LeakyReLU()
        self.siam=SiameseBlock(in_size)
      
      
        self.conv=nn.Conv2d(in_channels = self.dim//2 , out_channels = self.dim , kernel_size= 3,stride=1,padding=1)
        
        if cSE:
            self.cSE=ChannelSELayer(self.dim)
        else:
            self.cSE=lambda x:x
        

        
    def forward(self,x,y,lower_feature=None):
        x=self.siam(x)
        y=self.siam(y)
        
        
       
        z=torch.cat([x,y],dim=1)
        z=self.conv(z)
        z=self.cSE(z)
        z=self.lrelu(z)
        return z     

class GenGeneralblock(nn.Module):
    """ GenGeneralblock"""
    def __init__(self,in_size,cSE=True,Max=True,att=False):
        super(GenGeneralblock,self).__init__()
        self.in_size=in_size
        dim = dim_dict[in_size]
        if cSE :
            if Max:
                self.fusion=Fusionblock(in_size=in_size, cSE=True)
            else:
                self.fusion=NomaxFusionblock(in_size=in_size, cSE=True)
        else:
            if Max:
                self.fusion=Fusionblock(in_size=in_size, cSE=False)
            else:
                self.fusion=NomaxFusionblock(in_size=in_size, cSE=False)
        self.conv1 = nn.Conv2d(in_channels = dim , out_channels = dim , kernel_size= 3, padding=1)
        self.lrelu = nn.LeakyReLU()
        if att:
            
            self.att=Self_Attn(dim)
        else:
            self.att=nn.Identity()
        self.conv2=nn.Conv2d(in_channels = dim , out_channels = dim , kernel_size= 3, padding=1)
        
        
        
    def forward(self,x,y,lower=None):
        lower=nn.functional.interpolate(lower,scale_factor=2)
        temp=self.fusion(x,y,lower)
        out=self.conv1(temp)
        out=self.lrelu(out)
        out=self.att(out)
        out+=temp
        out=self.conv2(out)
        out=self.lrelu(out)
        
          
        return out

class InitGenGeneralblock(nn.Module):
    """ InitGenGeneralblock"""
    def __init__(self,in_size,cSE=True,Max=True):
        super(InitGenGeneralblock,self).__init__()
        self.in_size=in_size
        dim = dim_dict[in_size]
        if cSE :
            if Max:
                self.fusion=InitFusionblock(in_size=in_size, cSE=True)
            else:
                self.fusion=NomaxInitFusionblock(in_size=in_size, cSE=True)
        else:
            if Max:
                self.fusion=InitFusionblock(in_size=in_size, cSE=False)
            else:
                self.fusion=NomaxInitFusionblock(in_size=in_size, cSE=False)
        self.conv1 = nn.Conv2d(in_channels = dim , out_channels = dim , kernel_size= 3, padding=1)
        self.lrelu = nn.LeakyReLU()
        
        self.conv2=nn.Conv2d(in_channels = dim , out_channels = dim , kernel_size= 3, padding=1)
        
        
        
    def forward(self,x,y,lower=None):
        
        temp=self.fusion(x,y)
        out=self.conv1(temp)
        out=self.lrelu(out)
        out=self.conv2(out)
        out=self.lrelu(out)
        
          
        return out

class DisGeneralblock(nn.Module):
    """ DisGeneralblock"""
    def __init__(self,in_size,in_channels=3,init=False,att=False):
        super(DisGeneralblock,self).__init__()
        self.in_size=in_size
        dim = dim_dict[in_size]
        if init:
            in_channels=3
            scale=2
        else:
            last_dim=dim_dict[in_size*2]
            scale=dim//last_dim
            
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = dim//scale, kernel_size= 3, padding=1)
        self.lrelu = nn.LeakyReLU()
        if att:
            
            self.att=Self_Attn(dim//scale)
        else:
            self.att=nn.Identity()
        self.conv2=nn.Conv2d(in_channels = dim//scale , out_channels = dim , kernel_size= 3, padding=1)
        self.downsample=nn.AvgPool2d(2)
        
        
    def forward(self,x):
        
        out=self.conv1(x)
        out=self.lrelu(out)
        out=self.att(out)
        
        out=self.conv2(out)
        out=self.lrelu(out)
        out=self.downsample(out)
        
          
        return out
    
class DisFinalGeneralblock(nn.Module):
    """ DisFinalGeneralblock"""
    def __init__(self,in_size=4,in_channels=259):
        super(DisFinalGeneralblock,self).__init__()
        self.in_size=in_size
        dim = dim_dict[in_size]
        in_channels=3+dim_dict[in_size*2]
        
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = dim, kernel_size= 3, padding=1)
        self.lrelu = nn.LeakyReLU()
        self.att=Self_Attn(dim)
        if in_size==4:
            self.conv2=nn.Conv2d(in_channels = dim, out_channels = dim , kernel_size= 4, padding=0)
        else:
            self.conv2=nn.Sequential(nn.Conv2d(in_channels = dim, out_channels = dim , kernel_size= 3, padding=1,stride=2),
            nn.AdaptiveAvgPool2d((1,1)))                        
                                     
        
        
        # final conv layer emulates a fully connected layer
        self.conv3=nn.Conv2d(in_channels = dim, out_channels = 1 , kernel_size= 1, padding=0)
       
        
        
        
    def forward(self,x):
        
        out=self.conv1(x)
        out=self.lrelu(out)
        out=self.att(out)
        
        out=self.conv2(out)
        out=self.lrelu(out)
        out=self.conv3(out)
        
          
        return out.view(-1)