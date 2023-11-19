import zipfile
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def unzip_multi_focus(path):
    with zipfile.ZipFile(os.path.join(path, 'multi_focus.zip'), 'r') as zip_ref:
        zip_ref.extractall(path)

#unzip_multi_focus('/mnt/workspace/downloads/147292') unzip the file
class RawMultiFocusDataset(Dataset):#path is the basic path multi_focus
    def __init__(self, path,mode='train'):
        self.path = path
    
        image_filenames = os.listdir(os.path.join(path, 'image_jpg'))
        self.transform = transforms.Compose([
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        train_filenames, test_filenames = train_test_split(
                image_filenames, test_size=0.005, random_state=42)
        #split the data into train and test
        if mode=='train':
            self.image_filenames = train_filenames
        else:
            self.image_filenames = test_filenames
        
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        
        
        img_name = self.image_filenames[idx]
        image_path = os.path.join(self.path, 'image_jpg', img_name)
        a_path = os.path.join(self.path, 'A_jpg', f'A_{img_name}')
        b_path = os.path.join(self.path, 'B_jpg', f'B_{img_name}')

        image = Image.open(image_path).convert('RGB')
        a_image = Image.open(a_path).convert('RGB')
        b_image = Image.open(b_path).convert('RGB')
        
        
        
        # Apply the same transforms to all three images
        image = self.transform(image)
        a_image = self.transform(a_image)
        b_image = self.transform(b_image)
        
        return a_image, b_image, image





class MultiFocusDataset(Dataset):#path is the basic path multi_focus
    def __init__(self, path,mode='train',size=256):
        self.path = path
        self.size=size
        image_filenames = os.listdir(os.path.join(path, 'image_jpg'))
        self.transform = transforms.Compose([
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        train_filenames, test_filenames = train_test_split(
                image_filenames, test_size=0.005, random_state=42)
        #split the data into train and test
        if mode=='train':
            self.image_filenames = train_filenames
        else:
            self.image_filenames = test_filenames
        
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        
        
        img_name = self.image_filenames[idx]
        image_path = os.path.join(self.path, 'image_jpg', img_name)
        a_path = os.path.join(self.path, 'A_jpg', f'A_{img_name}')
        b_path = os.path.join(self.path, 'B_jpg', f'B_{img_name}')

        image = Image.open(image_path).convert('RGB')
        a_image = Image.open(a_path).convert('RGB')
        b_image = Image.open(b_path).convert('RGB')
        
        # Generate a random cropping position
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(self.size, self.size))
        
        # Apply the same cropping to all three images
        image = transforms.functional.crop(image, i, j, h, w)
        a_image = transforms.functional.crop(a_image, i, j, h, w)
        b_image = transforms.functional.crop(b_image, i, j, h, w)
        
        # Apply the same transforms to all three images
        image = self.transform(image)
        a_image = self.transform(a_image)
        b_image = self.transform(b_image)
        
        return a_image, b_image, image

    
def create_dataloader(dataset, batch_size, num_workers=0, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)




def test_multi_focus_dataset(dataset, num_images=2):  #show the image
    fig, ax = plt.subplots(num_images, 3, figsize=(8, 8))
    
    for i in range(num_images):
        a_image, b_image, image = dataset[i]
        
        ax[i][0].imshow(a_image.permute(1, 2, 0))
        ax[i][0].axis('off')
        ax[i][0].set_title('A image')
        
        ax[i][1].imshow(b_image.permute(1, 2, 0))
        ax[i][1].axis('off')
        ax[i][1].set_title('B image')
        
        ax[i][2].imshow(image.permute(1, 2, 0))
        ax[i][2].axis('off')
        ax[i][2].set_title('Input image')
        
    plt.show()
