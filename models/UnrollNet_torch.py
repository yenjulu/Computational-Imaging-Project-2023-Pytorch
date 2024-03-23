import torch
import torch.nn as nn
import data_consistency_torch as ssdu_dc
import models.networks_torch as networks
from models import unet
from utils import c2r, r2c
from utils_ssdu import torch_real2complex, torch_complex2real

class UnrolledNet(nn.Module):
    """
    Parameters
    ----------
    input_x: Tensor of shape (batch_size, nrow, ncol, 2) 
    sens_maps: Tensor of shape (batch_size, ncoil, nrow, ncol)
    trn_mask: Tensor of shape (batch_size, nrow, ncol), used in data consistency units
    loss_mask: Tensor of shape (batch_size, nrow, ncol), used to define loss in k-space

    args.nb_unroll_blocks: Number of unrolled blocks
    args.nb_res_blocks: Number of residual blocks in ResNet

    Returns
    ----------
    x: Nw output image
    nw_kspace_output: k-space corresponding new output at loss mask locations
    x0: dc output without any regularization.
    all_intermediate_results: All intermediate outputs of regularizer and dc units
    mu: Learned penalty parameter
    """

    def __init__(self, n_layers, k_iters):
        super(UnrolledNet, self).__init__()
        nb_res_blocks = n_layers
        self.nb_unroll_blocks = k_iters
        self.resnet =  networks.ResNet(nb_res_blocks) # used as a regularizer
        self.mu = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def forward(self, input_x, sens_maps, trn_mask, loss_mask):        
        dc_output = input_x.clone()                
  
        for i in range(self.nb_unroll_blocks):
   
            
            denoiser_output = self.resnet(dc_output)  # input for DC  shape (batch, nrow, ncol, 2)                                            
            rhs = input_x + self.mu * denoiser_output
            dc_output = ssdu_dc.dc_block(rhs, sens_maps, trn_mask, self.mu)  
                    
        nw_kspace_output = ssdu_dc.SSDU_kspace_transform(dc_output, sens_maps, loss_mask)
               
                
        return dc_output, nw_kspace_output


class UnrolledNet_GD(nn.Module):
    """
    Parameters
    ----------
    input_x: Tensor of shape (batch_size, nrow, ncol, 2) 
    sens_maps: Tensor of shape (batch_size, ncoil, nrow, ncol)
    trn_mask: Tensor of shape (batch_size, nrow, ncol), used in data consistency units
    loss_mask: Tensor of shape (batch_size, nrow, ncol), used to define loss in k-space

    args.nb_unroll_blocks: Number of unrolled blocks
    args.nb_res_blocks: Number of residual blocks in ResNet

    Returns
    ----------
    x: Nw output image
    nw_kspace_output: k-space corresponding new output at loss mask locations
    x0: dc output without any regularization.
    all_intermediate_results: All intermediate outputs of regularizer and dc units
    mu: Learned penalty parameter
    """

    def __init__(self, n_layers, k_iters):
        super().__init__()
        self.nb_unroll_blocks = k_iters
        self.mu = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.dc = ssdu_dc.data_consistency_ssdu_gd()
        self.dws = nn.ModuleList([unet.Unet(2, 2, num_pool_layers=n_layers) for _ in range(self.nb_unroll_blocks)])

    def forward(self, input_x, sens_maps, trn_mask, loss_mask):  
        #print('shape of input_x:', input_x.shape) # torch.Size([1, 396, 768, 2])      
        x0 = torch_real2complex(input_x) 
        #print('shape of x0:', x0.shape) # torch.Size([1, 396, 768])              
  
        for c in range(self.nb_unroll_blocks):
   
            if c == 0: 
                x_prev = x0.clone()
                          
            z = self.dc(x_prev, self.mu, x0, sens_maps, trn_mask)  # curr_x - self.lam * grad
            #print('shape of z:', z.shape)  # torch.Size([1, 396, 768])
            #print('shape of c2r(x_prev):', c2r(x_prev, axis=1).shape)  # torch.Size([1, 2, 396, 768])
            u = r2c(self.dws[c](c2r(x_prev, axis=1)), axis=1)
            x = z - u
            x_prev = x
            #print('shape of x_prev:', x_prev.shape)  # torch.Size([1, 396, 768])
                    
        nw_kspace_output = ssdu_dc.SSDU_kspace_transform_gd(x, sens_maps, loss_mask)
        #print('shape of nw_kspace_output:', nw_kspace_output.shape)  # torch.Size([1, 16, 396, 768])
                              
        return torch_complex2real(x), torch_complex2real(nw_kspace_output)