"""
Variational Network

Reference:
* Hammernik K, Klatzer T, Kobler E, Recht MP, Sodickson DK, Pock T, Knoll F. Learning a variational network for reconstruction of accelerated MRI data. Magn Reson Med 2018;79:3055-3071.

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple
import utils_ssdu
from utils import r2c, c2r
from models import mri, unet, transformer

# %%
class data_consistency_max_eig(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.lam = nn.Parameter(torch.tensor(1.), requires_grad=False)

    def get_max_eig(self, coil, mask, dcf=True):
        """ compute maximal eigenvalue

        References:
            * Beck A, Teboulle M.
              A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems.
              SIAM J Imaging Sci (2009). DOI: https://doi.org/10.1137/080716542
            * Tan Z, Hohage T, Kalentev O, Joseph AA, Wang X, Voit D, Merboldt KD, Frahm J.
              An eigenvalue approach for the automatic scaling of unknowns in model-based reconstructions: Application to real-time phase-contrast flow MRI.
              NMR Biomed (2017). DOI: https://doi.org/10.1002/nbm.3835
        """
        #A = mri.SenseOp(coil, mask, dcf=True)
        A = mri.SenseOp(coil, mask)

        device = coil.device
        x = torch.randn(size=mask.shape, dtype=coil.dtype, device=device)
        # print(x.shape) #[1, 384, 384]
        x = x / torch.linalg.norm(x)
        λ = 0
        tol=1e-12
        max_iter=30
        
        # for _ in range(30):
        #     y = A.adj(A.fwd(x))
        #     max_eig = torch.linalg.norm(y).ravel()
        #     x = y / max_eig

        #     # print(max_eig)

        for _ in range(max_iter):
            y = A.adj(A.fwd(x)) 
            y_norm = torch.linalg.norm(y)
            x = y / y_norm

            # Ax = A.adj(A.fwd(x))
            # λ = torch.dot(x.view(-1).T, Ax.view(-1))
            
            # err = torch.sum(torch.abs(Ax - λ * x))
            # if err < tol:
            #     break

        return y_norm  # λ

    def forward(self,
                curr_x: torch.Tensor,
                x0: torch.Tensor,
                coil: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:

        # A = mri.SenseOp(coil, mask, dcf=True,
        #                 device=coil.device)
        A = mri.SenseOp(coil, mask)

        self.max_eig = self.get_max_eig(coil, mask)
        #print(max_eig)
        
        grad = A.adj(A.fwd(curr_x) - x0)
        next_x = curr_x - (self.lam / self.max_eig) * grad

        return next_x

class data_consistency(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.lam = nn.Parameter(torch.tensor(0.01), requires_grad=True)

    def forward(self,
                curr_x: torch.Tensor,
                x0: torch.Tensor,
                coil: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:

        A = mri.SenseOp(coil, mask)
        grad = A.adj(A.fwd(curr_x)) - x0

        next_x = curr_x - self.lam * grad   # equation[3] in paper

        return next_x

# %%
class VarNet(nn.Module):
    def __init__(self, n_layers, k_iters) -> None:

        super().__init__()

        self.n_cascades = k_iters
        #############
        self.dc = data_consistency()
        #############
        self.dws = nn.ModuleList([unet.Unet(2, 2, num_pool_layers=n_layers) for _ in range(self.n_cascades)])
        

   
    def forward(self, x0, coil, mask):
        x0 = r2c(x0, axis=1)

        for c in range(self.n_cascades):
            if c == 0: 
                x_prev = x0.clone()
                
            z = self.dc(x_prev, x0, coil, mask)
            u = r2c(self.dws[c](c2r(x_prev, axis=1)), axis=1)
            x = z - u 
            x_prev = x

        return c2r(x, axis=1)
    

class VarNet_ssdu(nn.Module):
    def __init__(self, n_layers, k_iters) -> None:

        super().__init__()

        self.n_cascades = k_iters
        self.dc = data_consistency()
        self.dws = nn.ModuleList([unet.Unet(2, 2, num_pool_layers=n_layers) for _ in range(self.n_cascades)])
         
    def forward(self, x0, csm, trn_mask, loss_mask):
        x0 = r2c(x0, axis=1)

        for c in range(self.n_cascades):
            if c == 0: 
                x_prev = x0.clone()
                
            z = self.dc(x_prev, x0, csm, trn_mask)
            u = r2c(self.dws[c](c2r(x_prev, axis=1)), axis=1)
            x = z - u 
            x_prev = x
        
        kspace_x_k = self.SSDU_kspace(x, csm, loss_mask)
        
        return c2r(x, axis=1), c2r(kspace_x_k, axis=1)

    def SSDU_kspace(self, img, csm, loss_mask):
        """
        Transforms unrolled network output to k-space
        and selects only loss mask locations(\Lambda) for computing loss
        :img: zero-filled reconstruction (B, nrow, ncol) - complex64
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        :loss_mask: sampling mask (B, nrow, ncol) - int8               
        """

        csm = torch.swapaxes(csm, 0, 1)  # (coils, B, nrow, ncol)
        coil_imgs = csm * img
        #kspace = mri.fftc(coil_imgs, norm='ortho')
        kspace = utils_ssdu.fft_torch(coil_imgs, axes=(-2, -1), norm=None, unitary_opt=True)
        output = torch.swapaxes(loss_mask * kspace, 0, 1)

        return output  # B x coils x nrow x ncol    

# %%
class VarNet_Transformer(nn.Module):
    def __init__(self, n_layers, k_iters) -> None:

        super().__init__()

        self.n_cascades = k_iters
        self.dc = data_consistency()
        self.dws = nn.ModuleList([transformer.Transformer_full(n_layers) for _ in range(self.n_cascades)])
        
        
   
    def forward(self, x0, coil, mask):
        x0 = r2c(x0, axis=1)

        for c in range(self.n_cascades):
            if c == 0: 
                x_prev = x0.clone()
                
            z = self.dc(x_prev, x0, coil, mask)
            u = r2c(self.dws[c](c2r(x_prev, axis=1)), axis=1)
            x = z - u 
            x_prev = x

        return c2r(x, axis=1)
    
    
class VarNet_Unet_Transformer(nn.Module):
    def __init__(self, n_layers, k_iters) -> None:

        super().__init__()
        n_layers_tr = 1

        self.n_cascades = k_iters
        self.dc = data_consistency()       
        self.dws = nn.ModuleList([unet.Unet(2, 2, num_pool_layers=n_layers) for _ in range(self.n_cascades)])
        self.tr = nn.ModuleList([transformer.Transformer_full(n_layers_tr) for _ in range(self.n_cascades)])
        
   
    def forward(self, x0, coil, mask):
        x0 = r2c(x0, axis=1)

        for c in range(self.n_cascades):
            if c == 0: 
                x_prev = x0.clone()
                
            z = self.dc(x_prev, x0, coil, mask)
            u_inter = r2c(self.dws[c](c2r(x_prev, axis=1)), axis=1)
            u = r2c(self.tr[c](c2r(u_inter, axis=1)), axis=1)
            x = z - u 
            x_prev = x

        return c2r(x, axis=1)