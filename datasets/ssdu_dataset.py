from torch.utils.data import Dataset
import torch
import h5py as h5
import numpy as np
import utils_ssdu
import scipy.io as sio
from models import mri
import parser_ops
import random
from utils import r2c, c2r
#import sigpy as sp

parser = parser_ops.get_parser()
args = parser.parse_args()

def get_transformed_inputs_modl(args, kspace_train, sens_maps, trn_mask, loss_mask):
    
    kspace_train[:, :, :] = kspace_train[:, :, :] / np.max(np.abs(kspace_train[:, :, :][:]))
        
    sub_kspace = kspace_train * np.tile(trn_mask[np.newaxis, ...], (args.ncoil_GLOB, 1, 1)) 
    ref_kspace = kspace_train * np.tile(loss_mask[np.newaxis, ...], (args.ncoil_GLOB, 1, 1))
    #print("dtype of sub_kspace:", sub_kspace.dtype)
    nw_input = mri.sense1(sub_kspace, sens_maps).astype(np.complex64) #  nrow x ncol 
   
    ref_kspace = c2r(ref_kspace, axis=0)  #  2 x ncoil x nrow x ncol
    nw_input = c2r(nw_input, axis=0)  # 2 x row x col
       
    return ref_kspace, nw_input, sens_maps

def get_transformed_inputs_torch(args, kspace_train, sens_maps, trn_mask, loss_mask):
    
    kspace_train[:, :, :] = kspace_train[:, :, :] / np.max(np.abs(kspace_train[:, :, :][:]))

    nw_input = np.empty((args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
    ref_kspace = np.empty((args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
        
    sub_kspace = kspace_train * np.tile(trn_mask[..., np.newaxis], (1, 1, args.ncoil_GLOB)) 
    ref_kspace = kspace_train * np.tile(loss_mask[..., np.newaxis], (1, 1, args.ncoil_GLOB))
    #print("dtype of sub_kspace:", sub_kspace.dtype)
    nw_input = utils_ssdu.sense1(sub_kspace, sens_maps).astype(np.complex64) 

    # %% Prepare the data for the training
    sens_maps = np.transpose(sens_maps, (2, 0, 1))  # becomes : `ncoil` x `nrow` x `ncol`    
    ref_kspace = utils_ssdu.complex2real(np.transpose(ref_kspace, (2, 0, 1)))  # `ncoil` x `nrow` x `ncol` x 2
    nw_input = utils_ssdu.complex2real(nw_input)  # nrow` x `ncol` x 2
       
    return ref_kspace, nw_input, sens_maps

class ssdu_dataset_modl(Dataset):
    def __init__(self, mode, dataset_path, mask_path):
        """
        """
        if mode == 'train':
            self.prefix = 'trn'
        elif mode == 'val':
            self.prefix = 'tst'
        else:
            self.prefix = 'test' 
        
        self.dataset_path = dataset_path
        self.mask_path = mask_path

    def __getitem__(self, index):
        """
        """
        with h5.File(self.dataset_path, 'r') as f:
            gt, csm, kspace = f[self.prefix+'Org'][index], f[self.prefix+'Csm'][index], f[self.prefix+'Kspace'][index]
       
            gt = np.transpose(gt, axes=(1, 0))  # nrow x ncol            
            kspace = np.transpose(kspace, axes=(0, 2, 1)) # ncoil x nrow x ncol
            csm = np.transpose(csm, axes=(0, 2, 1))  # ncoil x nrow x ncol 
         
            max_gt = np.max(gt)
            min_gt = np.min(gt)
            # Normalize gt to 0-1
            gt = 1 * (gt - min_gt) / (max_gt - min_gt)
              
        with h5.File(self.mask_path, 'r') as f:
            num_masks = len(f['trn_mask'])
            mask_index = random.randint(0, num_masks - 1)
            
            trn_mask = f['trn_mask'][mask_index]
            loss_mask = f['loss_mask'][mask_index]
        
        if (self.prefix == 'test'):
            mask_dir = 'data/mask_poisson_accelx8_396_768.mat'
            input_mask = sio.loadmat(mask_dir)['mask']
            trn_mask = input_mask
            loss_mask = input_mask
            
        
        ref_kspace, nw_input, _  = get_transformed_inputs_modl(args, kspace, csm, trn_mask, loss_mask)

        return torch.from_numpy(gt), torch.from_numpy(ref_kspace), torch.from_numpy(nw_input), torch.from_numpy(csm), torch.from_numpy(trn_mask), torch.from_numpy(loss_mask)
        
    def __len__(self):
        with h5.File(self.dataset_path, 'r') as f:
            num_data = len(f[self.prefix+'Csm'])  
        return num_data

class ssdu_dataset_torch(Dataset):
    def __init__(self, mode, dataset_path, mask_path):
        """
        """
        if mode == 'train':
            self.prefix = 'trn'
        elif mode == 'val':
            self.prefix = 'tst'
        else:
            self.prefix = 'test'
        
        self.dataset_path = dataset_path
        self.mask_path = mask_path

    def __getitem__(self, index):
        """
        """
        with h5.File(self.dataset_path, 'r') as f:
            gt, csm, kspace = f[self.prefix+'Org'][index], f[self.prefix+'Csm'][index], f[self.prefix+'Kspace'][index]
         
            gt = np.transpose(gt, axes=(1, 0))  # nrow x ncol         
            kspace = np.transpose(kspace, axes=(2, 1, 0)) # nrow x ncol x ncoil
            csm = np.transpose(csm, axes=(2, 1, 0))  # nrow x ncol x ncoil   
         
            max_gt = np.max(gt)
            min_gt = np.min(gt)
            # Normalize gt to 0-1
            gt = 1 * (gt - min_gt) / (max_gt - min_gt)

        with h5.File(self.mask_path, 'r') as f:
            num_masks = len(f['trn_mask'])
            mask_index = random.randint(0, num_masks - 1)
            
            trn_mask = f['trn_mask'][mask_index]
            loss_mask = f['loss_mask'][mask_index]
        
        ref_kspace, nw_input, csm  = get_transformed_inputs_torch(args, kspace, csm, trn_mask, loss_mask)

        return torch.from_numpy(gt), torch.from_numpy(ref_kspace), torch.from_numpy(nw_input), torch.from_numpy(csm), torch.from_numpy(trn_mask), torch.from_numpy(loss_mask)
        
    def __len__(self):
        with h5.File(self.dataset_path, 'r') as f:
            num_data = len(f[self.prefix+'Csm'])  
        return num_data

