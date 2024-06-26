import torch
from torch.utils.data import Dataset
import h5py as h5
import numpy as np
from scipy.ndimage import gaussian_filter
import sys
import os
sys.path.append(os.path.abspath('../'))
from utils import * 
from models import mri
import math
import matplotlib.pyplot as plt
import sigpy as sp
import sigpy.mri as spmri
import scipy.io as sio


class modl_dataset(Dataset):
    def __init__(self, mode, dataset_path, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        self.prefix = 'trn' if mode == 'train' else 'tst'
        self.dataset_path = dataset_path
        self.sigma = sigma

    def __getitem__(self, index):
        """
        :x0: zero-filled reconstruction (2 x nrow x ncol) - float32
        :gt: fully-sampled image (2 x nrow x ncol) - float32
        :csm: coil sensitivity map (ncoil x nrow x ncol) - complex64
        :mask: undersample mask (nrow x ncol) - int8
        """
        with h5.File(self.dataset_path, 'r') as f:
            gt, csm, mask = f[self.prefix+'Org'][index], f[self.prefix+'Csm'][index], f[self.prefix+'Mask'][index]

        x0 = undersample_(gt, csm, mask, self.sigma)

        return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(csm), torch.from_numpy(mask)

    def __len__(self):
        with h5.File(self.dataset_path, 'r') as f:
            num_data = len(f[self.prefix+'Mask'])
        return num_data

class modl_fastmri_dataset(Dataset):
    def __init__(self, mode, dataset_path, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        if mode == 'train':
            self.prefix = 'trn'
        elif mode == 'val':
            self.prefix = 'tst'
        else:
            self.prefix = 'test'  
        
        self.dataset_path = dataset_path
        self.sigma = sigma
        
    def __getitem__(self, index):
        """
        :x0: zero-filled reconstruction (2 x nrow x ncol) - float32
        :gt: fully-sampled image (2 x nrow x ncol) - float32
        :csm: coil sensitivity map (ncoil x nrow x ncol) - complex64
        :mask: undersample mask (nrow x ncol) - int8
        """

        with h5.File(self.dataset_path, 'r') as f:

            gt, csm = f[self.prefix+'Org'][index], f[self.prefix+'Csm'][index]
            mask_dir = '/home/woody/rzku/mlvl125h/MoDL_PyTorch_v2023/data/mask_poisson_accelx8_384_384.mat'
            mask = sio.loadmat(mask_dir)['mask']
            mask = np.fft.ifftshift(mask, axes=(-2, -1))            

            gt = np.transpose(gt, axes=(1, 0))  # nrow x ncol            
            csm = np.transpose(csm, axes=(0, 2, 1))  # ncoil x nrow x ncol 
                    
            max_gt = np.max(gt)
            min_gt = np.min(gt)
            # Normalize gt to 0-1
            gt = 1 * (gt - min_gt) / (max_gt - min_gt)

        x0 = undersample_(gt, csm, mask, self.sigma)

        return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(csm), torch.from_numpy(mask)

    def __len__(self):
        with h5.File(self.dataset_path, 'r') as f:
            num_data = len(f[self.prefix+'Org'])
        return num_data

class modl_fastmri_datasetx4(Dataset):
    def __init__(self, mode, dataset_path, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        if mode == 'train':
            self.prefix = 'trn'
        elif mode == 'val':
            self.prefix = 'tst'
        else:
            self.prefix = 'test'  
        
        self.dataset_path = dataset_path
        self.sigma = sigma
        
    def __getitem__(self, index):
        """
        :x0: zero-filled reconstruction (2 x nrow x ncol) - float32
        :gt: fully-sampled image (2 x nrow x ncol) - float32
        :csm: coil sensitivity map (ncoil x nrow x ncol) - complex64
        :mask: undersample mask (nrow x ncol) - int8
        """

        with h5.File(self.dataset_path, 'r') as f:

            gt, csm = f[self.prefix+'Org'][index], f[self.prefix+'Csm'][index]
            mask_dir = '/home/woody/rzku/mlvl125h/MoDL_PyTorch_v2023/data/mask_poisson_accelx4_384_384.mat'
            mask = sio.loadmat(mask_dir)['mask']
            mask = np.fft.ifftshift(mask, axes=(-2, -1))            
        
            max_gt = np.max(gt)
            min_gt = np.min(gt)
            # Normalize gt to 0-1
            gt = 1 * (gt - min_gt) / (max_gt - min_gt)

        x0 = undersample_(gt, csm, mask, self.sigma)

        return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(csm), torch.from_numpy(mask)

    def __len__(self):
        with h5.File(self.dataset_path, 'r') as f:
            num_data = len(f[self.prefix+'Org'])
        return num_data

def undersample_(gt, csm, mask, sigma):

    ncoil, nrow, ncol = csm.shape
    csm = csm[None, ...]  # 4dim

    # shift sampling mask to k-space center
    mask = np.fft.ifftshift(mask, axes=(-2, -1))

    SenseOp = mri.SenseOp(csm, mask)

    b = SenseOp.fwd(gt)

    noise = torch.randn(b.shape) + 1j * torch.randn(b.shape)
    noise = noise * sigma / (2.**0.5)

    atb = SenseOp.adj(b + noise).squeeze(0).detach().numpy()

    return atb

def undersample(gt, csm, mask, sigma):
    """
    :get fully-sampled image, undersample in k-space and convert back to image domain
    """
    ncoil, nrow, ncol = csm.shape
    sample_idx = np.where(mask.flatten()!=0)[0]
    noise = np.random.randn(len(sample_idx)*ncoil) + 1j*np.random.randn(len(sample_idx)*ncoil)
    noise = noise * (sigma / np.sqrt(2.))
    b = piA(gt, csm, mask, nrow, ncol, ncoil) + noise #forward model
    atb = piAt(b, csm, mask, nrow, ncol, ncoil)
    return atb

def piA(im, csm, mask, nrow, ncol, ncoil):
    """
    fully-sampled image -> undersampled k-space
    """
    im = np.reshape(im, (nrow, ncol))
    im_coil = np.tile(im, [ncoil, 1, 1]) * csm #split coil images
    k_full = np.fft.fft2(im_coil, norm='ortho') #fft
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    k_u = k_full[mask!=0]
    return k_u

def piAt(b, csm, mask, nrow, ncol, ncoil):
    """
    k-space -> zero-filled reconstruction
    """
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    zero_filled = np.zeros((ncoil, nrow, ncol), dtype=np.complex64)
    zero_filled[mask!=0] = b #zero-filling
    img = np.fft.ifft2(zero_filled, norm='ortho') #ifft
    coil_combine = np.sum(img*csm.conj(), axis=0).astype(np.complex64) #coil combine
    return coil_combine
