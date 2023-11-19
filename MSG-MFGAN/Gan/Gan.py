import torch
import torch.nn as nn
import torch as th
from Gan.utils import *
from Gan.GeneralArc import *
import time
import timeit
from torch.utils.tensorboard import SummaryWriter
import torchvision
import datetime
import os



class Generator(nn.Module):
    """ MSG_MFGAN""" 
    def __init__(self,init_size=4,final_size=256,cSE=True,Max=True, use_spectral_norm=True):
        """_summary_

        Args:
            init_size (int, optional): _description_. Defaults to 4.the initial size of the input image
            final_size (int, optional): _description_. Defaults to 256.the final size of the output image
            cSE (bool, optional): _description_. Defaults to True.whether to use the channel-wise Squeeze-and-Excitation block in the fusion block
            Max (bool, optional): _description_. Defaults to True.whether to use the max combination in the fusion block
            use_spectral_norm (bool, optional): _description_. Defaults to True.whether to use the spectral normalization

        Returns:
            _type_: a list of the output images at various scales
        """
        super(Generator,self).__init__()
        self.spectral_norm_mode=None
        self.init_size=init_size
        self.final_size=final_size
        
        self.depth=int(np.log2(final_size/init_size))
        from torch.nn import ModuleList
        
        # register the modules required for the GAN Below ...
        # create the ToRGB layers for various outputs:
        def to_rgb(in_channels):
            return nn.Conv2d(in_channels, 3, (1, 1), bias=True)
        
        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([InitGenGeneralblock(self.init_size,cSE,Max)])
        self.rgb_converters = ModuleList([to_rgb(dim_dict[self.init_size])])
        
        # create the remaining layers
        for i in range(self.depth):
            
            layer = GenGeneralblock(self.init_size*np.power(2,i+1),cSE,Max)
            rgb = to_rgb(dim_dict[self.init_size*np.power(2,i+1)])
           
            self.layers.append(layer)
            self.rgb_converters.append(rgb)
            
        # if spectral normalization is on:
        if use_spectral_norm:
            self.turn_on_spectral_norm()
        
    def turn_on_spectral_norm(self):
        """
        private helper for turning on the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is False, \
                "can't apply spectral_norm. It is already applied"

        # apply the same to the remaining relevant blocks
        for module in self.layers:
            module.conv1 = spectral_norm(module.conv1)
            module.conv2 = spectral_norm(module.conv2)  
            
        # toggle the state variable:
        self.spectral_norm_mode = True  
        
    def turn_off_spectral_norm(self):
        """
        private helper for turning off the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import remove_spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is True, \
                "can't remove spectral_norm. It is not applied"

        # remove the applied spectral norm
        for module in self.layers:
            remove_spectral_norm(module.conv1)
            remove_spectral_norm(module.conv2)

        # toggle the state variable:
        self.spectral_norm_mode = False
            
    def forward(self,images):
        """
        forward pass of the Generator
        :param images: input images pairs  type:
        :return: *y => output of the generator at various scales
        """
        from torch import tanh
        outputs = []  # initialize to empty list

        y=1
        
        for block, converter, image_pair in zip(self.layers, self.rgb_converters,images):
            
            y = block(image_pair[0],image_pair[1],y)
            outputs.append(tanh(converter(y)))
            
        return outputs    

class Discriminator(nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, init_size=256,final_size=4,use_spectral_norm=True):
        """_summary_

        Args:
            init_size (int, optional): _description_. Defaults to 256.the initial size of the input image
            use_spectral_norm (bool, optional): _description_. Defaults to True.whether to use the spectral normalization

        Returns:
            _type_: a vector of the output of the discriminator
        """
        from torch.nn import ModuleList
        
        from torch.nn import Conv2d

        super(Discriminator,self).__init__()

        

        #calculate the depth of the discriminator
        self.depth =int(np.log2(init_size//final_size))
        
        self.spectral_norm_mode = None
        self.init_size = init_size

        

        

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([DisFinalGeneralblock(final_size)])

        # create the remaining layers
        for i in range(self.depth):
            if i < self.depth-1:
                layer = DisGeneralblock(final_size*(2**(i+1)),3+dim_dict[final_size*(2**(i+2))])
            else:
                layer = DisGeneralblock(final_size*(2**(i+1)),3)    
            self.layers.append(layer)
            

        # if spectral normalization is on:
        if use_spectral_norm:
            self.turn_on_spectral_norm()

    def turn_on_spectral_norm(self):
        """
        private helper for turning on the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is False, \
                "can't apply spectral_norm. It is already applied"

        # apply the same to the remaining relevant blocks
        for module in self.layers:
            module.conv1 = spectral_norm(module.conv1)
            module.conv2 = spectral_norm(module.conv2)

        # toggle the state variable:
        self.spectral_norm_mode = True

    def turn_off_spectral_norm(self):
        """
        private helper for turning off the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import remove_spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is True, \
                "can't remove spectral_norm. It is not applied"

        # remove the applied spectral norm
        for module in self.layers:
            remove_spectral_norm(module.conv1)
            remove_spectral_norm(module.conv2)

        # toggle the state variable:
        self.spectral_norm_mode = False

    def forward(self, inputs):
        """
        forward pass of the discriminator
        :param inputs: (multi-scale input images) to the network list[Tensors]
        :return: out => raw prediction values
        """
        
        assert len(inputs) == self.depth+1, \
            "Mismatch between input and Network scales" 
            

        
        y = self.layers[-1](inputs[-1])
        
        for x, block in \
                zip(reversed(inputs[:-1]),
                    reversed(self.layers[:-1])
                    ):
            
            y = torch.cat((x, y), dim=1)  # concatenate the inputs:
            y = block(y)  # apply the block
            
        return y





class MSG_MFGAN:
    """ MSG_MFGAN
        args:
            init_size: initial size of the image
            use_spectral_norm: whether to use spectral normalization to the convolutional
                               blocks.
            device: device to run the GAN on (GPU / CPU)
    """

    def __init__(self, init_size=4, final_size=256,
                 use_spectral_norm=False, device=torch.device("cpu"),data_parallel=False,cSE=True,Max=False):
        """ constructor for the class """
        if data_parallel:
            from torch.nn import DataParallel
        
        self.gen = Generator(init_size=init_size,final_size=final_size, use_spectral_norm=use_spectral_norm,cSE=cSE,Max=Max).to(device)
                            
        self.dis = Discriminator(init_size=final_size,final_size=init_size, use_spectral_norm=use_spectral_norm).to(device)
        
        # Create the Generator and the Discriminator
        if data_parallel:
            if device == th.device("cuda"):
                self.gen = DataParallel(self.gen)
                self.dis = DataParallel(self.dis)

        # state of the object
        #self.scaler = GradScaler()
        self.depth=int(np.log2(final_size/init_size))
        self.device = device
        self.init_size = init_size
        self.final_size = final_size
        st=int(np.log2(self.init_size))
      
        finish=int(np.log2(self.final_size))
        div_index=finish-st
        self.div_index=div_index
        # by default the generator and discriminator are in eval mode
        self.gen.eval()
        self.dis.eval()
    def load(self, gen_path=None, dis_path=None):
        """
        load the generator and discriminator from the given paths
        :param gen_path: path to the generator
        :param dis_path: path to the discriminator
        :return: None
        """
        if gen_path!=None:
            self.gen.load_state_dict(torch.load(gen_path))
        if dis_path!=None:
            self.dis.load_state_dict(torch.load(dis_path))
        return
    
    def generate_samples(self, num_samples,data,path="F:\dataset\multi_focus"):
        """
        generate samples using this gan
        :param num_samples: number of samples to be generated ,data: the interitive dataset to be used for the generation,path: the path of the multi_focus dataset
        :return: generated samples tensor: list[ Tensor(B x H x W x C)]
        """
        
        import sys 
        sys.path.append("..") 
        from data_loader import data_loader
        from torch.utils.data import Dataset, DataLoader
        
        #get data
        d=data_loader.MultiFocusDataset(path)
        d=DataLoader(d, batch_size=num_samples, shuffle=True)
        d=enumerate(d)
        #set the parameters
        init=self.init_size
        start=int(np.log2(init//4))
        final=self.final_size
        finish=int(np.log2(final))-2
        #create the muti-scale images list
        images=next(d)[1][0].to(self.device)
        i1=[images]+[nn.functional.avg_pool2d(images, int(np.power(2, i)))
                                            for i in range(1, 7)]
        images=next(d)[1][1].to(self.device)
        i2=[images]+[nn.functional.avg_pool2d(images, int(np.power(2, i)))
                                            for i in range(1, 7)]
        i=zip(i1,i2)
        #images pair in the form of list
        i=list(i)
        #reverse the i list
        i=i[::-1]
        #select the images pair in the range of start to finish
        i=i[start:finish+1]
        #generate the images
        generated_images = self.gen(i)

        # reshape the generated images
        generated_images = list(map(lambda x: (x.detach().permute(0, 2, 3, 1) / 2) + 0.5,
                                    generated_images))

        return generated_images
    def generate_samples2(self,imgArootPath,imgBrootPath=None,imgrootPath=None):
        """this func only for 256 size"""
        
        def getFilepairs(dir_path):
            """local function to get image pairs from a directory"""
           

            # get all filenames
            filenames = os.listdir(dir_path)

            # filter image filenames
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif','.tif']
            image_filenames = [filename for filename in filenames if any(filename.endswith(ext) for ext in image_extensions)]
            
            # get image pairs
            image_pairs = []
            for filename in image_filenames:
                    temp=filename.split('_')[-1]
                    suffix = temp.split('.')[-2]
                    if suffix == '1':
                        pair_filename = filename[:-5] + '2.'+temp.split('.')[-1]
                        if pair_filename in image_filenames:
                            image_pairs.append((filename, pair_filename))
            return image_pairs
        
        #get img name pair list
        
            
        image_filenames=getFilepairs(imgArootPath)
        
        for imgNameA, imgNameB in image_filenames:
            
            fullpathA=os.path.join(imgArootPath,imgNameA)
            fullpathB=os.path.join(imgBrootPath,imgNameB)
            resimg=self.merge_images(fullpathA,fullpathB,self.gen,div_index=self.div_index)
            
            imgName=imgNameA[:-5]+"proposed."+imgNameA.split('_')[-1].split('.')[-1]
            save_path = os.path.join(imgrootPath, imgName)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            resimg.save(save_path)
    
    
    def merge_images(self,A_path, B_path, model,div_index=6):
        import torchvision.transforms.functional as TF
        from PIL import Image
        import torchvision.transforms as transforms
        # loadA B
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        
        # pad A B to 256 's multiple
        w, h = A.size
        new_w = (w // 256 + 1) * 256
        new_h = (h // 256 + 1) * 256
        pad_A = TF.pad(A, (0, 0, new_w - w, new_h - h))
        pad_B = TF.pad(B, (0, 0, new_w - w, new_h - h))
        
        # crop A B to 256*256 subimages and fuse them
        transform = transforms.Compose([
                
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        subimages = []
        for i in range(0, new_h, 256):
            for j in range(0, new_w, 256):
                sub_A = TF.crop(pad_A, i, j, 256, 256)
                sub_B = TF.crop(pad_B, i, j, 256, 256)
                sub_A_tensor = transform(sub_A).unsqueeze(0).to(self.device)
                sub_B_tensor = transform(sub_B).unsqueeze(0).to(self.device)
                images=GetMuti_scale_Images_pair(sub_A_tensor,sub_B_tensor,div_index)
                sub_image_tensor = model(images)
                sub_image = sub_image_tensor[-1].squeeze(0)
                subimages.append(sub_image)
        
        # merge subimages to one image
        rows = []
        for i in range(0, len(subimages), new_w // 256):
            row = subimages[i:i+new_w//256]
            row=torch.cat(row,dim=2)
            rows.append(row)
            
        merged_image = torch.cat(rows, dim=1)
        merged_image = TF.crop(merged_image, 0, 0, h, w)
        merged_image = (merged_image + 1) / 2
        merged_image = TF.to_pil_image(merged_image)
        
        
        
        return merged_image

    def optimize_discriminator(self, dis_optim, multi_batch, real_batch, loss_fn):
        """
        performs one step of weight update on discriminator using the batch of data
        :param dis_optim: discriminator optimizer
        :param muti_batch: multi-scale muti-focus samples batch
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        
        fake_samples = self.gen(multi_batch)
        
        fake_samples = list(map(lambda x: x.detach(), fake_samples))

        loss = loss_fn.dis_loss(real_batch, fake_samples)
        
        # optimize discriminator
        dis_optim.zero_grad()
        loss.backward()
        dis_optim.step()
        # self.scaler.scale(loss).backward()
        # self.scaler.step(dis_optim)
        # self.scaler.update()

        return loss.item()

    def optimize_generator(self, gen_optim, multi_batch, real_batch, loss_fn):
        """
        performs one step of weight update on generator using the batch of data
        :param gen_optim: generator optimizer
        :param muti_batch: multi-scale muti-focus samples batch
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        
        fake_samples = self.gen(multi_batch)
            
        for i in range(len(fake_samples)):
            real_batch[i]=real_batch[i].to(fake_samples[0].dtype)
            
        loss,ssim = loss_fn.gen_loss(real_batch, fake_samples)

        # optimize discriminator
        gen_optim.zero_grad()
        loss.backward()
        gen_optim.step()
        # self.scaler.scale(loss).backward()
        # self.scaler.step(gen_optim)
        # self.scaler.update()

        return loss.item(),ssim.item()

    
    #@staticmethod
    def create_grid(samples, img_files):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing list[Tensors]
        :param img_files: list of names of files to write
        :return: None (saves multiple files)
        """
        from torchvision.utils import save_image
        from numpy import sqrt
        
        # save the images:
        for sample, img_file in zip(samples, img_files):
            sample = torch.clamp((sample.detach() / 2) + 0.5, min=0, max=1)
            save_image(sample, img_file, nrow=int(sqrt(sample.shape[0])))

    def train(self, data, gen_optim, dis_optim, loss_fn,testdata,
              start=1, num_epochs=12, feedback_factor=10, checkpoint_factor=1,
              data_percentage=100, num_samples=4,
              loss_dir='muti_recordings/loss', sample_dir='muti_recordings/images',
              save_dir="./models"):

        # TODOcomplete write the documentation for this method
        # no more procrastination ... HeHe
        """
        Method for training the network
        :param data: pytorch dataloader which iterates over images
        :param gen_optim: Optimizer for generator.
                          please wrap this inside a Scheduler if you want to
        :param dis_optim: Optimizer for discriminator.
                          please wrap this inside a Scheduler if you want to
        :param loss_fn: Object of GANLoss
        :param testdata: pytorch dataloader which iterates over test images
        :param start: starting epoch number
        :param num_epochs: total number of epochs to run for (ending epoch number)
                           note this is absolute and not relative to start
        :param feedback_factor: number of logs generated and samples generated
                                during training per epoch
        :param checkpoint_factor: save model after these many epochs
        :param data_percentage: amount of data to be used
        :param num_samples: number of samples to be drawn for feedback grid
        :param log_dir: path to directory for saving the loss.log file
        :param sample_dir: path to directory for saving generated samples' grids
        :param save_dir: path to directory for saving the trained models
        :return: None (writes multiple files to disk)
        """

        from torch.nn.functional import avg_pool2d
        writer_loss = SummaryWriter(loss_dir)
        writer_images=SummaryWriter(sample_dir)
            
       
        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()

        assert isinstance(gen_optim, th.optim.Optimizer), \
            "gen_optim is not an Optimizer"
        assert isinstance(dis_optim, th.optim.Optimizer), \
            "dis_optim is not an Optimizer"

        print("Starting the training process ... ")
        st=int(np.log2(self.init_size))
      
        finish=int(np.log2(self.final_size))
        div_index=finish-st
        self.div_index=div_index
        mid=int((st+finish)/2)
        
        # create fixed_input for debugging
        _,fixed=next(enumerate(testdata))
        fixed_inputA=fixed[0].to(self.device)
        fixed_inputB=fixed[1].to(self.device)
        
        fiexed_real=fixed[2].to(self.device)
        fixed_input=GetMuti_scale_Images_pair(imagesA=fixed_inputA,imagesB=fixed_inputB,div_index=div_index)
        fiexed_real=GetMuti_scale_Images(x=fiexed_real,div_index=div_index)
        
        image_grid_fakeA=torchvision.utils.make_grid(fixed_input[-1][0])
        writer_images.add_image("fixed-fakeA",image_grid_fakeA,1)
        image_grid_fakeB=torchvision.utils.make_grid(fixed_input[-1][1])
        writer_images.add_image("fixed-fakeB",image_grid_fakeB,1)
        
        image_grid_real=torchvision.utils.make_grid(fiexed_real[-1])
        writer_images.add_image("fixed-real",image_grid_real,1)
        len_of_img=len(fixed_input)
        # create a global time counter
        global_time = time.time()
        feedback_count=0
        for epoch in range(start, num_epochs + 1):
            start = timeit.default_timer()  # record time at the start of epoch

            print("\nEpoch: %d" % epoch)
            total_batches = len(iter(data))

            limit = int((data_percentage / 100) * total_batches)
            
            for (i, batch) in enumerate(data, 1):

                # extract current batch of data for training
                imagesA = batch[0].to(self.device)
                imagesB = batch[1].to(self.device)
                images  = batch[2].to(self.device)
                
                extracted_batch_size = images.shape[0]
                #calculate the range of the images
                
                
                # create list of downsampled images from the real images and muti-focus images:
                
                images =GetMuti_scale_Images(images,div_index)
                multi_focus =GetMuti_scale_Images_pair(imagesA,imagesB,div_index)

                
            
                

                # optimize the discriminator:
                dis_loss = self.optimize_discriminator(dis_optim, multi_focus,
                                                       images, loss_fn)

                # optimize the generator:
                # resample from the latent noise
                
                gen_loss,ssim_loss = self.optimize_generator(gen_optim, multi_focus,
                                                   images, loss_fn)

                # provide a loss feedback
                if i % int(limit / feedback_factor) == 0 or i == 1:
                    
                    with th.no_grad():
                        fixed_fake = self.gen(fixed_input)
                        ssim_test_loss=loss_fn.content_loss.SSIM(fixed_fake[-1],fiexed_real[-1])
                    
                    elapsed = time.time() - global_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [%s] feedback_count: %d batch: %d  d_loss: %f  g_loss: %f  ssim_train: %f  ssim_test: %f"
                          % (elapsed,feedback_count,i, dis_loss, gen_loss,ssim_loss, ssim_test_loss))
                    
                    # # also write the losses to the log file:
                    # if log_dir is not None:
                    #     log_file = os.path.join(log_dir, "loss.log")
                    #     os.makedirs(os.path.dirname(log_file), exist_ok=True)
                    #     with open(log_file, "a") as log:
                    #         log.write(str(dis_loss) + "\t" + str(gen_loss) +"\t"+ str(ssim_loss)+ "\n")
                    # write lossese to tensorboard
                    writer_loss.add_scalar('dis_loss', dis_loss, feedback_count)
                    writer_loss.add_scalar('gen_loss', gen_loss, feedback_count)
                    writer_loss.add_scalar('ssim_loss_train', ssim_loss, feedback_count)
                    writer_loss.add_scalar('ssim_loss_test', ssim_test_loss, feedback_count)

                    # create  grids of samples and save it
                    
                    feedback_count+=1
                    dis_optim.zero_grad()
                    gen_optim.zero_grad()
                    # with th.no_grad():
                    #     image_grid_max=torchvision.utils.make_grid(fixed_fake[-1]) 
                    #     image_grid_mid=torchvision.utils.make_grid(fixed_fake[len_of_img//2])
                    #     image_grid_min=torchvision.utils.make_grid(fixed_fake[0])
                    #     writer_images.add_image("fixed"+str(self.final_size),image_grid_max,feedback_count)
                    #     writer_images.add_image("fixed"+str(2**(mid)),image_grid_mid,feedback_count)
                    #     writer_images.add_image("fixed"+str(self.init_size),image_grid_min,feedback_count)
                        
                    
                
                if i > limit:
                    break

            # calculate the time required for the epoch
            stop = timeit.default_timer()
            print("Time taken for epoch: %.3f secs" % (stop - start))

            if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == num_epochs:
                os.makedirs(save_dir, exist_ok=True)
                gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(epoch) + ".pth")
                dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(epoch) + ".pth")
                gen_optim_save_file = os.path.join(save_dir,
                                                   "GAN_GEN_OPTIM_" + str(epoch) + ".pth")
                dis_optim_save_file = os.path.join(save_dir,
                                                   "GAN_DIS_OPTIM_" + str(epoch) + ".pth")

                th.save(self.gen.state_dict(), gen_save_file)
                th.save(self.dis.state_dict(), dis_save_file)
                th.save(gen_optim.state_dict(), gen_optim_save_file)
                th.save(dis_optim.state_dict(), dis_optim_save_file)
                # write the all scale images to tensorboard
                with th.no_grad():
                    for j in range(len_of_img):
                        img_grid=torchvision.utils.make_grid(fixed_fake[j])
                        writer_images.add_image("fixed"+str(2**j*self.init_size),img_grid,feedback_count)
                
                    rand_fake = self.gen(multi_focus)
                    #reduce the number of images batch in images list
                    rand_fake=[rand_fake[i][0:num_samples] for i in range(0,len(rand_fake))]
                    
                    for j in range(len_of_img):
                        img_grid=torchvision.utils.make_grid(rand_fake[j])
                        writer_images.add_image("rand"+str(2**j*self.init_size),img_grid,feedback_count)    

        print("Training completed ...")

        # return the generator and discriminator back to eval mode
        self.gen.eval()
        self.dis.eval()
# #simple code to test the Generator and the Discriminator

# #get data
# import sys 
# sys.path.append("..") 
# from data_loader import data_loader
# from torch.utils.data import Dataset, DataLoader
# d=data_loader.MultiFocusDataset("F:\dataset\multi_focus")
# d=DataLoader(d, batch_size=2, shuffle=True)
# d=enumerate(d)
# #set the parameters
# init=4
# start=int(np.log2(init//4))
# final=128
# finish=int(np.log2(final))-2
# #create the muti-scale images list
# images=next(d)[1][0]
# i1=[images]+[nn.functional.avg_pool2d(images, int(np.power(2, i)))
#                                      for i in range(1, 7)]
# images=next(d)[1][1]
# i2=[images]+[nn.functional.avg_pool2d(images, int(np.power(2, i)))
#                                      for i in range(1, 7)]
# i=zip(i1,i2)
# #images pair in the form of list
# i=list(i)
# #reverse the i list
# i=i[::-1]
# #select the images pair in the range of start to finish
# i=i[start:finish+1]   


# #test the Generator
# g=Generator(init_size=4,final_size=128,cSE=True,Max=True, use_spectral_norm=True)
# out=g(i)
# for x in out:
#     print(x.shape)

# #test the Discriminator

# d=Discriminator(init_size=final,use_spectral_norm=True)
# o=d(out)
# print(o.shape)

