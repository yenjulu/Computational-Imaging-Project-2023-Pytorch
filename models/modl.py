import torch
import torch.nn as nn

from utils import r2c, c2r
from models import mri, transformer, networks_torch
import utils_ssdu


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#CNN denoiser ======================
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class cnn_denoiser(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        layers = []
        layers += conv_block(2, 64)

        for _ in range(n_layers-2):  # n_layers : 1,2 will not execute
            layers += conv_block(64, 64)

        layers += nn.Sequential(
            nn.Conv2d(64, 2, 3, padding=1),
            nn.BatchNorm2d(2)
        )

        self.nw = nn.Sequential(*layers)

    def forward(self, x):
        idt = x # (2, nrow, ncol)
        dw = self.nw(x) + idt # (2, nrow, ncol)
        return dw

def fista(A, b, z_k, lam, num_iter=20, tol=1e-12):
    z_k = r2c(z_k, axis=1) # batch, nrow, ncol
    b = r2c(b, axis=1) # batch, nrow, ncol
    L = 1.0  # Step size, can be chosen or determined via line search
    x0 = torch.zeros_like(b)
    x = x0.clone()
    y = x0.clone()
    t = torch.tensor(1.0)

    for k in range(num_iter):
        x_old = x.clone()
        
        # Compute the gradient of the data fidelity term
        grad = A.adj(A.fwd(y) - b)
        
        # Perform a proximal step (soft thresholding) for the regularization term
        x = y - (1 / L) * grad
        x += -lam * (x - z_k)
        # Clamp the real and imaginary parts separately
        real = x.real.clamp(min=0)  # Clamp the real part
        imag = x.imag.clamp(min=0)  # Clamp the imaginary part (if needed)
        x = torch.complex(real, imag)  # Reconstruct the complex numbers

        
        # Update t and y for the next iteration
        t_old = t
        t = (1 + torch.sqrt(1 + 4 * t ** 2)) / 2
        y = x + ((t_old - 1) / t) * (x - x_old)
        
        if torch.norm(x - x_old) / torch.norm(x_old) < tol:
            break

    return c2r(x, axis=1)

### gmres alg.
def arnoldi_n(A, Q, P):
    # n-th step of arnoldi
    m, n = Q.shape
    dtype = Q.dtype
    device = Q.device  # Ensure compatibility with GPU tensors
    q = torch.zeros(m, dtype=dtype, device=device)
    h = torch.zeros(n + 1, dtype=dtype, device=device)
    # Compute the n-th Krylov vector
    v = torch.dot(A, Q[:, n-1])
    v = torch.linalg.solve_triangular(P, v, upper=True)
    
    # Arnoldi Iteration
    for j in range(n):
        h[j] = torch.dot(Q[:, j].conj(), v)
        v = v - h[j] * Q[:, j]

    h[n] = torch.linalg.norm(v)
    # Normalize if not zero
    if h[n] != 0:
        q = v / h[n]
    else:
        q = torch.zeros(m, dtype=dtype, device=device)
    return h, q

class Precondition(nn.Module):
    """
    """
    def __init__(self, csm, mask, lam):
        super(myAtA, self).__init__()
        self.csm = csm # complex (B x ncoil x nrow x ncol)
        self.mask = mask # complex (B x nrow x ncol)
        self.lam = lam
        self.A = mri.SenseOp(csm, mask)

    def forward(self, im): 
        """
        :im: complex image (B x nrow x nrol)
        """
        im_u = self.A.adj(self.A.fwd(im))   
        print(im_u.shape)
        P = np.diag(np.diag(im_u))
        print(P.shape)
        return P

def gmres_my(AtA, rhs, P=torch.eye(0), tol=1e-12):
    dtype = rhs.dtype
    device = rhs.device
    p = rhs.clone()
    
    AtA = AtA(p)   #print("shape of AtA:{}".format(AtA.shape))  # 1, 2, 384, 384
    AtA = r2c(AtA, axis=1)[0]  # 384, 384
    m = AtA.shape[0]  
    rhs = r2c(rhs, axis=1)[0] # nrow, ncol 
    # Reshape b into a vector
    rhs_vec = rhs.view(-1)  # 384*384

    
    if P.shape != A.shape:
        # default preconditioner P = I
        P = torch.eye(m, dtype=dtype, device=device)
    #x = np.zeros(m, dtype=b.dtype)
    x = torch.zeros_like(rhs)

    Q = torch.zeros((m, m+1), dtype=dtype, device=device)
    H = torch.zeros((m+1, m), dtype=dtype, device=device)
    
    b = torch.linalg.solve_triangular(P, b, upper=True)
    Q[:, 0] = b / torch.linalg.norm(b)
    b_hat = torch.zeros(m+1, dtype=dtype, device=device)
    b_hat[0] = torch.linalg.norm(b)

    for k in range(1, m+1):
        Qk = Q[:, :k]
        h, q = arnoldi_n(A, Qk, P)  # q: (m,)  h: (n+1,)  b: (m,) 

        H[:k+1, k-1] = h
        Q[:, k] = q

        Q_h, R_h = qr(H[:k+1, :k])
        y = torch.linalg.solve_triangular(R_h, torch.dot(Q_h[:k, :].T, b_hat[:k]), upper=True)
        
        x = torch.dot(Qk, y) 
        r_new = b - torch.linalg.solve_triangular(P, torch.dot(A, x), upper=True)
        relative_norm = torch.linalg.norm(r_new) / torch.linalg.norm(b)
        if relative_norm < tol:
            break
    return x

class data_consistency_gmres(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def forward(self, z_k, x0, csm, mask):
        rhs = x0 + self.lam * z_k # (2, nrow, ncol)   rhs = At*b + λz
        AtA = myAtA(csm, mask, self.lam)   
        rec = gmres_my(AtA, rhs)  # AtA(rec) = rhs
        return rec
#CG algorithm ======================
class myAtA(nn.Module):
    """
    performs DC step
    """
    def __init__(self, csm, mask, lam):
        super(myAtA, self).__init__()
        self.csm = csm # complex (B x ncoil x nrow x ncol)
        self.mask = mask # complex (B x nrow x ncol)
        self.lam = lam

        self.A = mri.SenseOp(csm, mask)

    def forward(self, im): #step for batch image
        """
        :im: complex image (B x nrow x nrol)
        """
        im_u = self.A.adj(self.A.fwd(im))    # im_u = A*A(x)
        return im_u + self.lam * im   # A*A(x) + λx = AtA(x)

def myCG(AtA, rhs):  # inputs : A , b => A: AtA, b: rhs
    """
    performs CG algorithm
    :AtA: a class object that contains csm, mask and lambda and operates forward model
    """
    rhs = r2c(rhs, axis=1) # nrow, ncol
    x = torch.zeros_like(rhs)
    i, r, p = 0, rhs, rhs   # r =rhs - AtA(x) = rhs.  
    rTr = torch.sum(r.conj()*r).real
    while i < 10 and rTr > 1e-10:           # i < 10
        Ap = AtA(p)  # AtA() 
        alpha = rTr / torch.sum(p.conj()*Ap).real
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(r.conj()*r).real
        beta = rTrNew / rTr
        p = r + beta * p
        i += 1
        rTr = rTrNew
    return c2r(x, axis=1)

class data_consistency_fista(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def forward(self, z_k, x0, csm, mask):
        A = mri.SenseOp(csm, mask)

        rec = fista(A, x0, z_k, self.lam)  # x0, zk : (batch, 2, nrow, ncol)
    
        return rec

class data_consistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def forward(self, z_k, x0, csm, mask):
        rhs = x0 + self.lam * z_k # (2, nrow, ncol)   rhs = At*b + λz
        AtA = myAtA(csm, mask, self.lam)   
        rec = myCG(AtA, rhs)  # AtA(rec) = rhs
        return rec

class data_consistency_gd(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.lam = nn.Parameter(torch.tensor(0.01), requires_grad=True)

    def forward(self,
                curr_x: torch.Tensor,
                z_k: torch.Tensor,
                x0: torch.Tensor,
                coil: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:

        A = mri.SenseOp(coil, mask)
        curr_x, x0, z_k = r2c(curr_x, axis=1), r2c(x0, axis=1), r2c(z_k, axis=1)

        # print(coil.shape)
        # print(curr_x.shape)
        # print(x0.shape)
        # print(z_k.shape)
        # torch.Size([4, 16, 384, 384])
        # torch.Size([4, 2, 384, 384])
        # torch.Size([4, 2, 384, 384])
        # torch.Size([4, 2, 384, 384])
        
        ### GD1
        grad = A.adj(A.fwd(curr_x)) - x0 + self.lam * (curr_x - z_k)
        next_x = curr_x - grad   # equation[3] in paper
        
        ### GD2
        # grad = self.lam * (A.adj(A.fwd(curr_x)) - x0) + (curr_x - z_k)
        # next_x = curr_x - grad   # equation[3] in paper        

        return c2r(next_x, axis=1)

#model =======================
class MoDL_gd(nn.Module):
    def __init__(self, n_layers, k_iters):
        """
        :n_layers: number of layers
        :k_iters: number of iterations
        """
        super().__init__()
        self.k_iters = k_iters
        self.dw = cnn_denoiser(n_layers)
        self.dc = data_consistency_gd()  ########
        ##############
        # self.dws = nn.ModuleList([cnn_denoiser(n_layers) for _ in range(k_iters)])
        ##############        

    def forward(self, x0, csm, mask):
        """
        :x0: zero-filled reconstruction (B, 2, nrow, ncol) - float32
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        :mask: sampling mask (B, nrow, ncol) - int8
        """

        x_k = x0.clone()
        for k in range(self.k_iters):
            # cnn denoiser
            z_k = self.dw(x_k) # (2, nrow, ncol)
            #####################
            # z_k = self.dws[k](x_k)
            #####################            
            # data consistency
            x_k = self.dc(x_k, z_k, x0, csm, mask)  # (2, nrow, ncol)
        return x_k

class MoDL(nn.Module):
    def __init__(self, n_layers, k_iters):
        """
        :n_layers: number of layers
        :k_iters: number of iterations
        """
        super().__init__()
        self.k_iters = k_iters
        self.dw = cnn_denoiser(n_layers)
        #############
        self.dc = data_consistency()
        ##############
        # self.dws = nn.ModuleList([cnn_denoiser(n_layers) for _ in range(k_iters)])
        ##############        

    def forward(self, x0, csm, mask):
        """
        :x0: zero-filled reconstruction (B, 2, nrow, ncol) - float32
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        :mask: sampling mask (B, nrow, ncol) - int8
        """

        x_k = x0.clone()
        for k in range(self.k_iters):
            # cnn denoiser
            z_k = self.dw(x_k) # (2, nrow, ncol)
            #####################
            # z_k = self.dws[k](x_k)
            #####################            
            # data consistency
            x_k = self.dc(z_k, x0, csm, mask) # (2, nrow, ncol)
        return x_k

class MoDL_Transformer(nn.Module):
    def __init__(self, n_layers, k_iters):
        """
        :n_layers: number of layers
        :k_iters: number of iterations
        """
        super().__init__()                
        self.k_iters = k_iters
        self.dc = data_consistency()
        self.tr = transformer.Transformer_full(n_layers)
        # self.tr = t2t_vit.T2T_ViT(tokens_type='performer', embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., **kwargs)
        # self.trs = nn.ModuleList([t2t_vit.T2T_ViT(tokens_type='transformer', embed_dim=768, depth=n_layers, num_heads=12, mlp_ratio=2.).to(device) for _ in range(k_iters)])
        ##############
        #self.trs = nn.ModuleList([transformer.Transformer_full(n_layers) for _ in range(k_iters)])
        ############## 

    def forward(self, x0, csm, mask):
        """
        :x0: zero-filled reconstruction (B, 2, nrow, ncol) - float32
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        :mask: sampling mask (B, nrow, ncol) - int8
        """

        x_k = x0.clone()
        for k in range(self.k_iters):
            # cnn denoiser
            z_k = self.tr(x_k) # (2, nrow, ncol)
            #####################
            # z_k = self.trs[k](x_k)
            #####################             
            # data consistency
            x_k = self.dc(z_k, x0, csm, mask) # (2, nrow, ncol)
        return x_k

class MoDL_ssdu(nn.Module):
    def __init__(self, n_layers, k_iters):
        """
        :n_layers: number of layers
        :k_iters: number of iterations
        """
        super().__init__()
        self.k_iters = k_iters
        #########
        self.dw = cnn_denoiser(n_layers)
        # self.dw = networks_torch.ResNet_modl(n_layers)
        # self.dw = transformer.Transformer_full(n_layers)
        #########
        self.dc = data_consistency()
       
    def forward(self, x0, csm, trn_mask, loss_mask):
        """
        :x0: zero-filled reconstruction (B, 2, nrow, ncol) - float32
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        :mask: sampling mask (B, nrow, ncol) - int8
        """

        x_k = x0.clone()
        for k in range(self.k_iters):
            # cnn denoiser
            z_k = self.dw(x_k) # (B, 2, nrow, ncol)
                      
            # data consistency
            x_k = self.dc(z_k, x0, csm, trn_mask) # (B, 2, nrow, ncol)
            
        kspace_x_k = self.SSDU_kspace(x_k, csm, loss_mask)
            
        return x_k, kspace_x_k

    def SSDU_kspace(self, img, csm, loss_mask):
        """
        Transforms unrolled network output to k-space
        and selects only loss mask locations(\Lambda) for computing loss
        :img: zero-filled reconstruction (B, 2, nrow, ncol) - float32
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        :loss_mask: sampling mask (B, nrow, ncol) - int8               
        """
        img = r2c(img, axis=1) # (B, 2, nrow, ncol)  ---> (B, nrow, ncol)
        csm = torch.swapaxes(csm, 0, 1)  # (coils, B, nrow, ncol)
        coil_imgs = csm * img
        
        #kspace = mri.fftc(coil_imgs, axes=(-2, -1), norm='ortho')
        kspace = utils_ssdu.fft_torch(coil_imgs, axes=(-2, -1), norm=None, unitary_opt=True)
        
        output = torch.swapaxes(loss_mask * kspace, 0, 1)

        return c2r(output, axis=1)  # B x 2 x coils x nrow x ncol
    
