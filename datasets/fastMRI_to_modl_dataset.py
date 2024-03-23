import torch
from torch.utils.data import Dataset
import scipy.io as sio
import h5py
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../'))
from utils import * 
from models import mri
import matplotlib.pyplot as plt
import sigpy as sp
from sigpy.mri import samp, app

def generate_fastMRI_tst_dataset(source_dir='../data/brain/multicoil_test', output_file='../data/fastmri_tst_dataset_ssdu.hdf5'):
    '''
    this dataset contains same coil numbers = 16
    '''  
    filenames_dir = '../data/filenames_test.mat'
    test_filenames = sio.loadmat(filenames_dir)['test_filenames']
    
    # Open the HDF5 file in 'a' append mode
    with h5py.File(output_file, 'a') as h5_combined:
            
        # Process testing files
        for filename in test_filenames:
            file_path = os.path.join(source_dir, filename)

            try:
                Csm, Org, kspace  = fastMRI_to_modl_dataset_ssdu(file_path)
                Csm = Csm[:, :16, :, :]
                kspace = kspace[:, :16, :, :]

                # Iterate over each dataset name and corresponding data
                for name, data in zip(['testCsm', 'testOrg', 'testKspace'],
                                      [Csm, Org, kspace]):
                    if name in h5_combined:
                        # Dataset exists, so resize it to fit the new data
                        dataset = h5_combined[name]
                        current_size = dataset.shape[0]
                        new_size = current_size + data.shape[0]
                        dataset.resize(new_size, axis=0)
                        # Append the new data
                        dataset[current_size:] = data
                    else:
                        # Dataset does not exist, so create it
                        if name == 'testOrg':
                            h5_combined.create_dataset(name, data=data, maxshape=(None, 768, 396), chunks=(1, 768, 396))
                        else:
                            h5_combined.create_dataset(name, data=data, maxshape=(None, 16, 768, 396), chunks=(1, 16, 768, 396))                         

            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")

def generate_fastMRI_h5_dataset(source_dir='../data/brain/multicoil_train', output_file='../data/fastmri_dataset_ssdu.hdf5'):
    '''
    this dataset contains same coil numbers = 16
    '''  
    filenames_dir = '../data/filenames.mat'
    train_filenames = sio.loadmat(filenames_dir)['train_filenames']
    val_filenames = sio.loadmat(filenames_dir)['val_filenames'] 
    
    # Open the HDF5 file in 'a' append mode
    with h5py.File(output_file, 'a') as h5_combined:
        
        for filename in train_filenames:
            file_path = os.path.join(source_dir, filename)

            try:
                Csm, Org, kspace = fastMRI_to_modl_dataset_ssdu(file_path)
                Csm = Csm[:, :16, :, :]
                kspace = kspace[:, :16, :, :]

                # Iterate over each dataset name and corresponding data
                for name, data in zip(['trnCsm', 'trnOrg', 'trnKspace'],
                                      [Csm, Org, kspace]):
                    if name in h5_combined:
                        # Dataset exists, so resize it to fit the new data
                        dataset = h5_combined[name]
                        current_size = dataset.shape[0]
                        new_size = current_size + data.shape[0]
                        dataset.resize(new_size, axis=0)
                        # Append the new data
                        dataset[current_size:] = data
                    else:
                        # Dataset does not exist, so create it
                        # The maxshape parameter is set to allow resizing later
                        #chunk_size = (data.shape[0],) + data.shape[1:]
                        if name == 'trnCsm':
                            h5_combined.create_dataset(name, data=data, maxshape=(None, 16, 768, 396), chunks=(1, 16, 768, 396))
                        elif name == 'trnKspace':
                            h5_combined.create_dataset(name, data=data, maxshape=(None, 16, 768, 396), chunks=(1, 16, 768, 396))                       
                        else:
                            h5_combined.create_dataset(name, data=data, maxshape=(None, 768, 396), chunks=(1, 768, 396))    

            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")
    
        # Process testing files
        for filename in val_filenames:
            file_path = os.path.join(source_dir, filename)

            try:
                Csm, Org, kspace  = fastMRI_to_modl_dataset_ssdu(file_path)
                Csm = Csm[:, :16, :, :]
                kspace = kspace[:, :16, :, :]

                # Iterate over each dataset name and corresponding data
                for name, data in zip(['tstCsm', 'tstOrg', 'tstKspace'],
                                      [Csm, Org, kspace]):
                    if name in h5_combined:
                        # Dataset exists, so resize it to fit the new data
                        dataset = h5_combined[name]
                        current_size = dataset.shape[0]
                        new_size = current_size + data.shape[0]
                        dataset.resize(new_size, axis=0)
                        # Append the new data
                        dataset[current_size:] = data
                    else:
                        # Dataset does not exist, so create it
                        # The maxshape parameter is set to allow resizing later
                        if name == 'tstCsm':
                            h5_combined.create_dataset(name, data=data, maxshape=(None, 16, 768, 396), chunks=(1, 16, 768, 396))
                        elif name == 'tstKspace':
                            h5_combined.create_dataset(name, data=data, maxshape=(None, 16, 768, 396), chunks=(1, 16, 768, 396))                       
                        else:
                            h5_combined.create_dataset(name, data=data, maxshape=(None, 768, 396), chunks=(1, 768, 396))    


            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")

def fastMRI_to_modl_dataset(dataset_path):
    with h5py.File(dataset_path, 'r') as file:
      kspace = file['kspace'][:]   # <class 'numpy.ndarray'>

    N_slice, N_coil, _, _ = kspace.shape
       
    recon_rss_1 = sp.rss(sp.ifft(kspace, axes=[-2, -1]), axes=(-3))
    Org = sp.resize(recon_rss_1, oshape=[N_slice, 384, 384])   

    device = sp.Device(0) if torch.cuda.is_available() else sp.cpu_device
    kspace_dev = sp.to_device(kspace, device=device)
    csm = []
    for s in range(N_slice):
        k = kspace_dev[s]
        c = app.EspiritCalib(k, device=device).run()
        c = sp.to_device(c)
        c = sp.resize(c, oshape=[N_coil, 384, 384])
        csm.append(c)

    Csm = np.array(csm)   
    return Csm, Org, kspace

def fastMRI_to_modl_dataset_ssdu(dataset_path):
    with h5py.File(dataset_path, 'r') as file:
      kspace = file['kspace'][:]   # <class 'numpy.ndarray'>

    N_slice, N_coil, _, _ = kspace.shape
       
    recon_rss_1 = sp.rss(sp.ifft(kspace, axes=[-2, -1]), axes=(-3))
    Org = recon_rss_1
    # Org = sp.resize(recon_rss_1, oshape=[N_slice, 384, 384])   

    device = sp.Device(0) if torch.cuda.is_available() else sp.cpu_device
    kspace_dev = sp.to_device(kspace, device=device)
    csm = []
    for s in range(N_slice):
        k = kspace_dev[s]
        c = app.EspiritCalib(k, device=device).run()
        c = sp.to_device(c)
        # c = sp.resize(c, oshape=[N_coil, 768, 396])
        csm.append(c)

    Csm = np.array(csm)   
    return Csm, Org, kspace

def gen_mask():
    
    from scipy.io import savemat
    #mask_poisson = samp.poisson([256, 232], 12).astype(np.int8)
    mask_poisson = samp.poisson([396, 768], 8).astype(np.int8) 
    savemat('../data/mask_poisson_accelx8_396_768.mat', {'mask': mask_poisson})
    # mask_poisson = samp.poisson([384, 384], 12).astype(np.int8) 
    # savemat('../data/mask_poisson_accelx12_384_384.mat', {'mask': mask_poisson})

def gen_trn_loss_mask(output_file='../data/trn_loss_mask_ssdu_2.hdf5'):
    from models.ssdu_masks import ssdu_masks
    ssdumask = ssdu_masks()

    with h5py.File(output_file, 'a') as h5_combined:
        for _ in range(2600):
            try:
                trn_mask, loss_mask = ssdumask.Gaussian_selection()
                trn_mask, loss_mask  = trn_mask[None, ...], loss_mask[None, ...]
                #print(trn_mask.shape) # (1, 396, 768)

                # Iterate over each dataset name and corresponding data
                for name, data in zip(['trn_mask', 'loss_mask'],
                                      [trn_mask, loss_mask]):
                    if name in h5_combined:
                        # Dataset exists, so resize it to fit the new data
                        dataset = h5_combined[name]
                        current_size = dataset.shape[0]
                        new_size = current_size + data.shape[0]
                        dataset.resize(new_size, axis=0)
                        # Append the new data
                        dataset[current_size:] = data
                    else:
                        # Dataset does not exist, so create it    
                        h5_combined.create_dataset(name, data=data, maxshape=(None, 396, 768), chunks=(1, 396, 768))    

            except Exception as e:
                print(f"An error occurred while processing : {e}")    

    

def get_dataset_info(source_dir='../data/brain/multicoil_test'):
    
    coil_sizes = []
    xy_sizes = []

    for f in os.listdir(source_dir):
        if f.startswith('file_brain_AXT2_210_6001'):
            file_path = os.path.join(source_dir, f)
            with h5py.File(file_path, 'r') as hdf_file:
                _, c, x, y = hdf_file['kspace'].shape
                coil_sizes.append(c)
                xy_sizes.append((x, y))

    unique_coil, counts_coil = np.unique(coil_sizes, return_counts=True)
    unique_xy, counts_xy = np.unique(xy_sizes, return_counts=True)
    
    output_coil = (dict(zip(unique_coil, counts_coil)))
    output_xy = (dict(zip(unique_xy, counts_xy)))
    
    return output_coil, output_xy

if __name__ == "__main__":
    # generate_fastMRI_h5_dataset()
    generate_fastMRI_tst_dataset()

    # gen_trn_loss_mask()
    # gen_mask()
    # output_coil, output_xy = get_dataset_info()
    