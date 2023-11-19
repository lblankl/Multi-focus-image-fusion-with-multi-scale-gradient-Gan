from torch import nn
import numpy as np
def GetMuti_scale_Images(x, div_index):
        """
        :param x: input image
        :param start: start scale
        :param finish: finish scale
        :return: list of images at different scales
        """
        # list of images at different scales
        
        images=[nn.functional.avg_pool2d(x, int(np.power(2, i)))
                                            for i in range(div_index,-1 ,-1)]
        
        
        
        return images

def GetMuti_scale_Images_pair(imagesA,imagesB, div_index):
        """_summary_

        :param imagesA: image A's patch
        :param imagesB: image B's patch
        :param start: start scale
        :param finish: finish scale
        :return: list of images pair at different scales
        
        """
        imagesA=[nn.functional.avg_pool2d(imagesA, int(np.power(2, i)))
                                            for i in range(div_index,-1 ,-1)]
        imagesB=[nn.functional.avg_pool2d(imagesB, int(np.power(2, i)))
                                            for i in range(div_index,-1,-1)]
        multi_focus=zip(imagesA,imagesB)
        multi_focus=list(multi_focus)
        
        return multi_focus
        