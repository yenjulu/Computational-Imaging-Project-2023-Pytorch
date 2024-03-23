Introduction to the Project:

(1) Generating the appropriate input data from multi-coil brain fastmri MRI k-space source data for model processing.
(2) Evaluating the MoDL framework, a PyTorch-implemented Model-Based Deep Learning Architecture for Inverse Problems.
Assessing the Varnet framework, another PyTorch implementation aimed at Learning a Variational Network for the Reconstruction of Accelerated MRI Data.
(3) Transforming neural networks for self-supervised learning and reconstruction without fully sampled reference data, implemented in TensorFlow, into a PyTorch implementation. Furthermore, evaluating the SSDU framework.

Conducting comprehensive evaluations that include:
(a) Variations in the number of unrolled models.
(b) The use of convolutional layers in RestNet or encoder-decoder layer configurations in Unet.
(c) The performance across three types of source data: T1-weighted, T2-weighted, and FLAIR images.
(d) The substitution of traditional Unet or ResNet structures with a Vision Transformer architecture.
(e) The performance of models under 4x, 8x, and 12x acceleration masks.

# MoDL

Official code: https://github.com/hkaggarwal/modl

![image](https://github.com/yenjulu/MoDL_and_Varnet_PyTorch/assets/126617563/3cae4de9-5bed-449a-99ff-e492b911e897)

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

## Saved models

Saved models are provided.

K=1, N=9: `workspace/fastmri_modl,k=1,n=9/checkpoints/best.epoch0017-score32.5437.pth` 

# Varnet

![image](https://github.com/yenjulu/MoDL_and_Varnet_PyTorch/assets/126617563/ed10edd3-f318-4455-b064-83e51fb4159e)

## Reference paper

Learning a Variational Network for Reconstruction of Accelerated MRI Data by Kerstin Hammernik, Teresa Klatzer, Erich Kobler, Michael P. Recht, Daniel K. Sodickson, Thomas Pock, Florian Knoll in Magnetic Resonance in Medicine, 2018

## Dataset ## Configuration file ## Train ## Test

The same path as described in MoDL

## Saved models

Saved models are provided.

K=6, N=3: `workspace/fastmri_varnet,k=6,n=3/checkpoints/best.epoch0043-score38.0277.pth`

# SSDU

Official code:  https://github.com/byaman14/SSDU (TensorFlow)

![image](https://github.com/yenjulu/MoDL_Varnet_SSDU_PyTorch/assets/126617563/963f14b8-3e09-41fb-a3f5-2ae4014610be)

## Reference paper

Self-supervised learning of physics-guided reconstruction neural networks without fully sampled reference data by Burhaneddin Yaman, Seyed Amir Hossein Hosseini, Steen Moeller, Jutta Ellermann, Kâmil Uğurbil, Mehmet Akçakaya, 2020

## Dataset ## Configuration file ## Train ## Test

The same path as described in MoDL

## Saved models

Saved models are provided.

K=6, N=3: `workspace/fastmri_ssdu,k=6,n=3/checkpoints/best.epoch0043-score38.0277.pth`

