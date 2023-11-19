import torch as th
from pytorch_msssim import MS_SSIM

# TODO_complete Major rewrite: change the interface to use only predictions
# for real and fake samples
# The interface doesn't need to change to only use predictions for real and fake samples
# because for loss such as WGAN-GP requires the samples to calculate gradient penalty
class ContentLoss:
    
    def __init__(self, device, dis):
        self.device = device
        self.dis = dis
        self.MS_SSIM = MS_SSIM(data_range=1, size_average=True, channel=3,win_size=7)
    def SSIM(self, real_samps, fake_samps):
        """
        Calculate the structural similarity measure index
        :param real_samps: real samples.Note that original real samples are list of muti-scale images.Input is the last one of list. pixel values are in range [-1,1] 
        you need to scale them to [0,1] before calculating the SSIM loss
        :param fake_samps: fake samples.Same as real samples
        :return: the SSIM loss
        """
        
        #calculate the pixel values in range [0,1]
        real_samps = (real_samps + 1) / 2
        fake_samps = (fake_samps + 1) / 2
        
        # calculate the SSIM loss
        return 1 -self.MS_SSIM(real_samps, fake_samps)
    
    def L1(self, real_samps, fake_samps):
        """
        Calculate the L1 loss
        :param real_samps: real samples.Note that original real samples are list of muti-scale images.
        :param fake_samps: fake samples.Same as real samples
        :return: the L1 loss
        """
        return th.mean(th.abs(real_samps - fake_samps))


class GANLoss:
    """
    Base class for all losses
    Note that the gen_loss also has
    """

    def __init__(self, device, dis):
        self.device = device
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps):
        raise NotImplementedError("gen_loss method has not been implemented")

    def conditional_dis_loss(self, real_samps, fake_samps, conditional_vectors):
        raise NotImplementedError("conditional_dis_loss method has not been implemented")

    def conditional_gen_loss(self, real_samps, fake_samps, conditional_vectors):
        raise NotImplementedError("conditional_gen_loss method has not been implemented")

class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, device, dis):
        super().__init__(device, dis)

    def dis_loss(self, real_samps, fake_samps):
        # difference between real and fake:
        r_f_diff = self.dis(real_samps) - th.mean(self.dis(fake_samps))

        # difference between fake and real samples
        f_r_diff = self.dis(fake_samps) - th.mean(self.dis(real_samps))

        # return the loss
        return (th.mean(th.nn.ReLU()(1 - r_f_diff))
                + th.mean(th.nn.ReLU()(1 + f_r_diff)))

    def gen_loss(self, real_samps, fake_samps):
        # difference between real and fake:
        r_f_diff = self.dis(real_samps) - th.mean(self.dis(fake_samps))

        # difference between fake and real samples
        f_r_diff = self.dis(fake_samps) - th.mean(self.dis(real_samps))

        # return the loss
        return (th.mean(th.nn.ReLU()(1 + r_f_diff))
                + th.mean(th.nn.ReLU()(1 - f_r_diff)))
class LSGAN(GANLoss):

    def __init__(self, device, dis):
        super().__init__(device, dis)

    def dis_loss(self, real_samps, fake_samps):
        return 0.5 * (((th.mean(self.dis(real_samps)) - 1) ** 2)
                      + (th.mean(self.dis(fake_samps))) ** 2)

    def gen_loss(self, _, fake_samps):
        return 0.5 * ((th.mean(self.dis(fake_samps)) - 1) ** 2)

class RelativeplusContentLoss(GANLoss):

    def __init__(self, device, dis,alpha=3,beta=2):
        super().__init__(device, dis)
        self.content_loss = ContentLoss(device, dis)
        #
        self.relative_loss = LSGAN(device, dis)
        #self.relative_loss=RelativisticAverageHingeGAN(device, dis)
        self.alpha=alpha
        self.beta=beta
    def dis_loss(self, real_samps, fake_samps):
        return self.relative_loss.dis_loss(real_samps, fake_samps)

    def gen_loss(self, real_samps, fake_samps):
        
        l1=0
        rescaled_ssims=0
        for i in range(len(real_samps)):
            
            l=self.alpha*self.content_loss.L1(real_samps[i], fake_samps[i])
            l1+=l
        
        ssim=self.content_loss.SSIM(real_samps[-2], fake_samps[-2])
        rescaled_ssims+=ssim*self.beta
        ssim=self.content_loss.SSIM(real_samps[-1], fake_samps[-1])
        rescaled_ssims+=ssim*self.beta
        rela_gan=self.relative_loss.gen_loss(real_samps, fake_samps)
        
        
        return  rescaled_ssims+ rela_gan+l1,ssim