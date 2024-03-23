# open tensorboard  
tensorboard --logdir=./  --port=6013
# SBATCH --output=train_test_k=1_%j.txt
# scp -J mlvl125h@cshpc.rrze.fau.de .\dataset.hdf5 mlvl125h@tinyx.nhr.fau.de:/home/woody/rzku/mlvl125h/MoDL_PyTorch-master
# scp -J mlvl125h@cshpc.rrze.fau.de mlvl125h@tinyx.nhr.fau.de:<remoteFilePath> <localDirectory>

# salloc --partition=rtx3080 --gres=gpu:1 --time=04:00:00
# salloc --partition=a100 --gres=gpu:a100:1 --time=23:00:00
:x0: zero-filled reconstruction (2 x nrow x ncol) - float32
:gt: fully-sampled image (2 x nrow x ncol) - float32
:csm: coil sensitivity map (ncoil x nrow x ncol) - complex64
:mask: undersample mask (nrow x ncol) - int8

# kspace         Shape: (16, 20, 768, 396), Type: complex64
# rss            Shape: (16, 384, 384),     Type: float32

# Keys: ['gt', 'recon']
# Shape: (32, 256, 232), Type: float32
# Shape: (32, 256, 232), Type: float32

# filename = 'dataset.hdf5'
# Keys: ['trnCsm', 'trnMask', 'trnOrg', 'tstCsm', 'tstMask', 'tstOrg']
# shape as : slices, coils, rows, columns
 
# Exploring trnCsm  Shape: (360, 12, 256, 232), Type: complex64     
# Exploring trnMask Shape: (360, 256, 232), Type: int8
# Exploring trnOrg  Shape: (360, 256, 232), Type: complex64
# Exploring tstCsm  Shape: (164, 12, 256, 232), Type: complex64
# Exploring tstMask Shape: (164, 256, 232), Type: int8
# Exploring tstOrg  Shape: (164, 256, 232), Type: complex64

<!-- git add .
git add path/to/your/file
git add path/to/your/folder/
git commit -m "20240323 update files"
git push -u origin master

git init
git remote add origin https://github.com/yenjulu/MoDL_PyTorch.git

upload files:
git add datasets/modl_dataset.py
git add models
git add get_instances.py
git add test.py
git add train.py
git add utils.py -->


Gaussian selection is processing, rho = 0.40, center of kspace: center-kx: 198, center-ky: 384
 Gaussian selection is processing, rho = 0.40, center of kspace: center-kx: 198, center-ky: 384
  Gaussian selection is processing, rho = 0.40, center of kspace: center-kx: 198, center-ky: 383