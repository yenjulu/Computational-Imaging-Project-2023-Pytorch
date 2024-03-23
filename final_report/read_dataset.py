import h5py
import matplotlib
matplotlib.use('Agg')  # Or another backend suitable for your system
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../'))
from utils import psnr, ssim
from models.mri import fftc, ifftc
import math
import torch
import torch.nn as nn
from torchsummary import summary
import sigpy as sp

def get_image(index, filename):
    with h5py.File(filename, 'r') as file:

        image_slice_gt = file['gt'][index, :, :]
        image_slice_recon = file['recon'][index, :, :]
        #image_slice_recon = file['recon'][index, 6:6+384, 192:192+384]
        return image_slice_recon, image_slice_gt

def plot_images_comparison(image_list, saved_name, save_dir='./plot'):
    import math

    plt.figure(figsize=(16, 8)) 
    plt.subplot(1, 1, 1)
    
    y_pred = image_list[0]
    y_pred = np.rot90(y_pred, k=-1)
    y = image_list[1]
    y = np.rot90(y, k=-1)    

    plt.imshow(np.abs(y_pred-y), cmap='gray', vmin=np.abs(y).min(), vmax=np.abs(y).max())
    # plt.title(f"{name}", fontsize = 26)
    plt.axis('off') 
    
    plt.savefig(f'{save_dir}/{saved_name}.png', dpi=1000)
    plt.close()
        
def plot_images(image_list, saved_name, save_dir='./plot'):
    import math
    num_images = len(image_list)
    rows = 2
    cols = math.ceil(num_images / rows)
    plt.figure(figsize=(16, 8)) 
    
    for i, (image) in enumerate(image_list):
            
        plt.subplot(rows, cols, i + 1)
        magnitude = np.abs(image)
        magnitude = np.rot90(magnitude, k=-1)

        plt.imshow(magnitude, cmap='gray')
        # plt.title(f"{name}", fontsize = 26)
        plt.axis('off') 
    
    plt.savefig(f'{save_dir}/{saved_name}.png', dpi=1000)
    plt.close()

def content_of_dataset(dataset_path):
    with h5py.File(dataset_path, 'r') as f:
        index = 0
        gt, csm, kspace = f['trnOrg'][index], f['trnCsm'][index], f['trnKspace'][index]

        gt = np.transpose(gt, axes=(1, 0))  # nrow x ncol
        kspace = np.transpose(kspace, axes=(2, 1, 0)) # nrow x ncol x ncoil
        csm = np.transpose(csm, axes=(2, 1, 0))  # nrow x ncol x ncoil           
            
        max_gt = np.max(gt)
        min_gt = np.min(gt)
        # Normalize gt to 0-1
        gt = 1 * (gt - min_gt) / (max_gt - min_gt)  
                  
        return kspace, gt, csm

def get_model_state_dict():
 # Load the file as a dictionary
    checkpoint  = torch.load(filename)

    # Inspect the keys and values
    for key in checkpoint.keys():
        print(f"Layer in the model: {key}")

    optim_state_dict = checkpoint['optim_state_dict']
    model_state_dict = checkpoint['model_state_dict']

    # Now iterate over the 'model_state_dict' to print what's inside
    for key, value in optim_state_dict.items():
        print(f"Layer: {key}")
        print(f"parameters: {value}")
        print(f"Shape of parameters: {value.shape}")
        print(f"Type of parameters: {type(value)}")

def model_parameters():
    from models.modl import MoDL, MoDL_Transformer
    from models.varnet import VarNet, VarNet_Transformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_layers = 1 
    k_iters = 5    
    model = MoDL_Transformer(n_layers, k_iters)
    model.to(device)
    total_params = 0

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            num_params = parameter.numel()
            total_params += num_params
            print(f"{name} has {num_params} parameters")

    print(f"Total trainable parameters: {total_params}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_coil_img_sizes():
    coil_sizes = []
    img_sizes = []

    unique, counts = np.unique(img_sizes, return_counts=True)
    print(dict(zip(unique, counts)))
    
    unique, counts = np.unique(coil_sizes, return_counts=True)
    print(dict(zip(unique, counts))) # {8: 2, 12: 1, 16: 88, 18: 2, 20: 61}

def file_keys():
    filename = '../data/fastmri_tst_dataset.hdf5'  
    filename = '../runs/fastmri_modl,k=1,n=4/test/recon_001.h5'


    with h5py.File(filename, 'r') as file:
        keys = list(file.keys())
        print("Keys: %s" % keys)

        for key in keys:
            dataset = file[key]
            print(f"Shape: {dataset.shape}, Type: {dataset.dtype}")  

 

saved_name = 'recon_000'  # 2389

img_list = []
img1_list = []
img2_list = []

i = 2
for i in range(10)    
    filename = '../runs/fastmri_ssdu,k=6,n=12,modl/test/'f'recon_00{i}.h5'               
    recon, gt = get_image(0, filename)
    img_list.append(recon)
# max_recon = np.max(recon)
# min_recon = np.min(recon)
# recon = 1 * (recon - min_recon) / (max_recon - min_recon)
    
# psnr_value = psnr(gt, recon)
# print(np.abs(gt).max(), np.abs(gt).min())
# print(np.abs(recon).max(), np.abs(recon).min())
# print(psnr_value)

# img_list.append(gt)

plot_images(img_list, saved_name='ssdu_modl_train_imgs', save_dir='./plot') 
# plot_images_comparison(img_list, saved_name='ssdu_modl_train_imgs_compare', save_dir='./plot')

