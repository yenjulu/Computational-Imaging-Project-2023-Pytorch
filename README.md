Introduction to the Project:

(1) Generating the appropriate input data from multi-coil brain fastmri MRI k-space source data for model processing.
(2) Evaluating the MoDL framework, a PyTorch-implemented Model-Based Deep Learning Architecture for Inverse Problems.
Assessing the Varnet framework, another PyTorch implementation aimed at Learning a Variational Network for the Reconstruction of Accelerated MRI Data.
(3) Transforming neural networks for self-supervised learning and reconstruction without fully sampled reference data, implemented in TensorFlow, into a PyTorch implementation, furthermore, evaluating the SSDU framework.

Conducting comprehensive evaluations that include:
(a) Variations in the number of unrolled-block in models.
(b) The use of different numbers of convolutional layers in RestNet or encoder-decoder layer configurations in Unet.
(c) Implementation of the SSDU concept with CNN-based and ResNet-based models.
(d) Substituting traditional Unet or ResNet structures with a Vision Transformer architecture.
(e) The performance of models under 4x, 8x, and 12x acceleration masks.

# MoDL

Official code: https://github.com/hkaggarwal/modl

![MoDL_model](https://github.com/yenjulu/Computational-Imaging-Project-2023-Pytorch/assets/126617563/fe83ca94-3b3d-4aba-a016-32e9b1174157)

![MoDL_psnr_K](https://github.com/yenjulu/Computational-Imaging-Project-2023-Pytorch/assets/126617563/e9f18ac6-f14c-4e39-ae03-9a7765548aa3)

![MoDL_psnr_N](https://github.com/yenjulu/Computational-Imaging-Project-2023-Pytorch/assets/126617563/83f1a2fa-2601-49dd-b877-56f184b9d4b5)

## Reference paper

MoDL: Model Based Deep Learning Architecture for Inverse Problems  by H.K. Aggarwal, M.P Mani, and Mathews Jacob in IEEE Transactions on Medical Imaging,  2018 

Link: https://arxiv.org/abs/1712.02862

IEEE Xplore: https://ieeexplore.ieee.org/document/8434321/

## Dataset

The multi-coil brain fastmri dataset used in this project is different from the one used in the original paper.

**Link** : https://fastmri.med.nyu.edu/

## Configuration file

The configuration files are in `config` folder. 

## Train

You can change the configuration file for training by modifying the `train.sh` file.

```
scripts/train.sh
```

## Test

You can change the configuration file for testing by modifying the `test.sh` file.

```
scripts/test.sh
```

## Best configuration of model

K=1, N=9: `workspace/fastmri_modl,k=1,n=9/log.txt` 

# Varnet

![Varnet_model](https://github.com/yenjulu/Computational-Imaging-Project-2023-Pytorch/assets/126617563/bd6e8c55-0b0c-4518-bea6-f015e3594bf7)

![Varnet_psnr](https://github.com/yenjulu/Computational-Imaging-Project-2023-Pytorch/assets/126617563/3707a315-b806-416b-bce8-b42049f71a2a)

## Reference paper

Learning a Variational Network for Reconstruction of Accelerated MRI Data by Kerstin Hammernik, Teresa Klatzer, Erich Kobler, Michael P. Recht, Daniel K. Sodickson, Thomas Pock, Florian Knoll in Magnetic Resonance in Medicine, 2018

## Dataset ## Configuration file ## Train ## Test

The same path as described in MoDL

## Best configuration of model

K=6, N=3: `workspace/fastmri_varnet,k=6,n=3/log.txt`

# SSDU

Official code:  https://github.com/byaman14/SSDU (TensorFlow)

![SSDU_model](https://github.com/yenjulu/Computational-Imaging-Project-2023-Pytorch/assets/126617563/8170f1c2-18fc-4054-a8a7-17419f19f6ea)


## Reference paper

Self-supervised learning of physics-guided reconstruction neural networks without fully sampled reference data by Burhaneddin Yaman, Seyed Amir Hossein Hosseini, Steen Moeller, Jutta Ellermann, Kâmil Uğurbil, Mehmet Akçakaya, 2020

## Dataset ## Configuration file ## Train ## Test

The same path as described in MoDL

## Best configuration of model

K=5, N=6: `workspace/fastmri_ssdu,k=5,n=6,modl,resnet,ispace/log.txt`

# Transformer

![MoDL_Transformer_psnr](https://github.com/yenjulu/Computational-Imaging-Project-2023-Pytorch/assets/126617563/295092c0-030e-4841-906e-1167a3d131a7)

![Varnet_Transformer_psnr](https://github.com/yenjulu/Computational-Imaging-Project-2023-Pytorch/assets/126617563/955a12f4-178b-4634-956b-dedca0a68e6d)

# Comparison tables and trainable parameter counts

<img width="310" alt="table parameter counts" src="https://github.com/yenjulu/Computational-Imaging-Project-2023-Pytorch/assets/126617563/449845f9-7611-4783-865b-79b2d8c24acf">

![compare tables](https://github.com/yenjulu/Computational-Imaging-Project-2023-Pytorch/assets/126617563/d549323b-e529-4f0b-9281-d9f30bfa33ec)

# Recontrcucted images

![img_modl_varnet](https://github.com/yenjulu/Computational-Imaging-Project-2023-Pytorch/assets/126617563/707b9c98-e9e0-4695-a94e-82c4c612710d)

![img_ssdu](https://github.com/yenjulu/Computational-Imaging-Project-2023-Pytorch/assets/126617563/76b54c46-843c-4d8b-822b-1516f04317ed)
