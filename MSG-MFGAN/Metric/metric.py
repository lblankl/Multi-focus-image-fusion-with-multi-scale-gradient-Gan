import torch as th

from sklearn import metrics
from pytorch_msssim import MS_SSIM ,SSIM
from ennemi import mutual_information

class MetricForMultiFusion:
    def __init__(self,device,gen):
        self.gen=gen
        self.device=device
        
        self.MS_SSIM = MS_SSIM(data_range=1, size_average=True, channel=3,win_size=11)
        self.SSIM = SSIM(data_range=1, size_average=True, channel=3)
    def MS_SSIM(self, real_samps, fake_samps):
        """
        Calculate the MS structural similarity measure index
        :param real_samps: real samples.Note that original real samples are list of muti-scale images.Input is the last one of list. pixel values are in range [-1,1] 
        you need to scale them to [0,1] before calculating 
        :param fake_samps: fake samples.Same as real samples
        :return: the MS_SSIM 
        """
        #calculate the pixel values in range [0,1]
        real_samps = (real_samps + 1) / 2
        fake_samps = (fake_samps + 1) / 2
        
        # calculate the SSIM loss
        return self.MS_SSIM(real_samps, fake_samps)
    
    def SSIM(self, real_samps, fake_samps):
        """
        Calculate the structural similarity measure index
        :param real_samps: real samples.Note that original real samples are list of muti-scale images.Input is the last one of list. pixel values are in range [-1,1] 
        you need to scale them to [0,1] before calculating 
        :param fake_samps: fake samples.Same as real samples
        :return: the SSIM 
        """
        #calculate the pixel values in range [0,1]
        real_samps = (real_samps + 1) / 2
        fake_samps = (fake_samps + 1) / 2
        
        # calculate the SSIM loss
        return self.SSIM(real_samps, fake_samps)
    