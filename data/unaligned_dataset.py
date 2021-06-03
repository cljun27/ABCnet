import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import sys
import SimpleITK as sitk
import numpy as np
import h5py
import torch


    
class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """


    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # BaseDataset.__init__(self, opt)
        self.opt = opt
        self.dir_img = os.path.join(opt.dataroot, opt.phase + '/img')  # create a path '/path/to/data/trainA'
        self.dir_bf = os.path.join(opt.dataroot, opt.phase + '/bf')  # create a path '/path/to/data/trainB'
        self.dir_wm = os.path.join(opt.dataroot, opt.phase + '/wm')  # create a path '/path/to/data/trainB'
        self.dir_gm = os.path.join(opt.dataroot, opt.phase + '/gm')  # create a path '/path/to/data/trainB'
        self.dir_min_img = os.path.join(opt.dataroot, opt.phase + '/img_min')  # create a path '/path/to/data/trainB'
        self.dir_max_img = os.path.join(opt.dataroot, opt.phase + '/img_max')  # create a path '/path/to/data/trainB'
        self.dir_min_bf = os.path.join(opt.dataroot, opt.phase + '/bf_min')  # create a path '/path/to/data/trainB'
        self.dir_max_bf = os.path.join(opt.dataroot, opt.phase + '/bf_max')  # create a path '/path/to/data/trainB'
        # self.loader_img = h5_img_slides_loader
        # self.loader_bf = h5_bf_slides_loader
        # print(self.dir_B)
        self.img_paths = sorted(make_dataset(self.dir_img, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.bf_paths = sorted(make_dataset(self.dir_bf, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.img_size = len(self.img_paths)  # get the size of dataset A
        self.bf_size = len(self.B_pbf_pathsaths)  # get the size of dataset B
        self.wm_paths = sorted(make_dataset(self.dir_wm, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.gm_paths = sorted(make_dataset(self.dir_gm, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.wm_size = len(self.wm_paths)  # get the size of dataset A
        self.gm_size = len(self.gm_paths)  # get the size of dataset B
        self.min_img_paths = sorted(make_dataset(self.dir_min_img, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.max_img_paths = sorted(make_dataset(self.dir_max_img, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.min_bf_paths = sorted(make_dataset(self.dir_min_bf, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.max_bf_paths = sorted(make_dataset(self.dir_max_bf, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.min_img_size = len(self.min_img_paths)  # get the size of dataset A
        self.max_img_size = len(self.max_img_paths)  # get the size of dataset B        
        self.min_bf_size = len(self.min_bf_paths)  # get the size of dataset A
        self.max_bf_size = len(self.max_bf_paths)  # get the size of dataset B        
        # print(self.B_size)
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        # self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        # self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        img_path = self.img_paths[index % self.img_size]  # make sure index is within then range
        # print k('A', A_path)
        # if self.opt.serial_batches:   # make sure index is within then range
        #     index_B = index % self.B_size
        # else:   # randomize the index for domain B to avoid fixed pairs.
        #     index_B = random.randint(0, self.B_size - 1)
        # index_B = index % self.B_size
        bf_path = self.bf_paths[index % self.bf_size]  # make sure index is within then range
        wm_path = self.wm_paths[index % self.wm_size]  # make sure index is within then range
        gm_path = self.gm_paths[index % self.gm_size]  # make sure index is within then range
        min_img_path = self.min_img_paths[index % self.min_img_size]  # make sure index is within then range
        max_img_path = self.max_img_paths[index % self.max_img_size]  # make sure index is within then range
        min_bf_path = self.min_bf_paths[index % self.min_bf_size]  # make sure index is within then range
        max_bf_path = self.max_bf_paths[index % self.max_bf_size]  # make sure index is within then range
        # B_path = self.B_paths[index_B]
        # print('B',B_path)
        ori_img = np.load(img_path)
        bf_img = np.load(bf_path)
        wm_img = np.load(wm_path)
        gm_img = np.load(gm_path)
        min_ori = np.load(min_img_path)
        max_ori = np.load(max_img_path)        
        min_bf = np.load(min_bf_path)
        max_bf = np.load(max_bf_path)   

        ori_img = torch.from_numpy(ori_img)
        bf_img = torch.from_numpy(bf_img)
        wm_img = torch.from_numpy(wm_img)
        gm_img = torch.from_numpy(gm_img)
        min_ori_norm = torch.from_numpy(min_ori)
        max_ori_norm = torch.from_numpy(max_ori)
        min_bf_norm = torch.from_numpy(min_bf)
        max_bf_norm = torch.from_numpy(max_bf)
        ori_img = ori_img.unsqueeze(0)
        bf_img = bf_img.unsqueeze(0)
        wm_img = wm_img.unsqueeze(0)
        gm_img = gm_img.unsqueeze(0)
        min_ori_norm = min_ori_norm.unsqueeze(0)
        max_ori_norm = max_ori_norm.unsqueeze(0)
        min_bf_norm = min_bf_norm.unsqueeze(0)
        max_bf_norm = max_bf_norm.unsqueeze(0)
        # print('A.shape',A_img.shape)
        # print('B.shape',B_img.shape)

        # B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        # A = self.transform_A(A_img)
        # B = self.transform_B(B_img)
        Ori = ori_img
        # print(A.shape)
        BF = bf_img

        WM = wm_img
        GM = gm_img

        

        return {'Ori': Ori, 'BF': BF, 'img_paths': img_path, 'bf_paths': bf_path, 'WM': WM, 'GM': GM, 'wm_paths': wm_path, 'gm_paths': gm_path, 'min_ori_norm': min_ori_norm, 'max_ori_norm': max_ori_norm, 'min_img_paths': min_img_path, 'max_img_paths': maxA_path, 'min_bf_norm': min_bf_norm, 'max_bf_norm': max_bf_norm, 'min_bf_paths': min_bf_path, 'max_bf_paths': max_bf_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.img_size, self.bf_size)
