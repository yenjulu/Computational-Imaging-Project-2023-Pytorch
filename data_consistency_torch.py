import torch
import torch.nn as nn
import utils_ssdu as torch_utils

### finished to be torch version ###
class data_consistency_ssdu_gd(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, img, mu, x0, sens_maps, mask):
        """
        Performs x - λ * [(E^h*E) x - x0]
        """
        shape_list = mask.shape
        scalar = torch.sqrt(torch.tensor(shape_list[-2] * shape_list[-1], dtype=torch.float32))
        scalar = torch.complex(scalar, torch.zeros_like(scalar))
                
        # E: operation of img -> coil_img -> masked-kspace coil_img  
        coil_imgs = sens_maps * img  # # batch x 16 x nrow x ncol
        kspace = torch_utils.torch_fftshift(torch.fft.fftn(torch_utils.torch_ifftshift(coil_imgs),dim=(-2, -1))) / scalar
        masked_kspace = kspace * mask  # batch x 16 x nrow x ncol

        # Eh: operation of masked-kspace coil_img -> spatial-space coil_img * scalar -> times conj of sens_maps
        image_space_coil_imgs = torch_utils.torch_ifftshift(torch.fft.ifftn(torch_utils.torch_fftshift(masked_kspace),dim=(-2, -1))) * scalar
        image_space_comb = torch.sum(image_space_coil_imgs * torch.conj(sens_maps), dim=1) #dim=0  # batch x nrow x ncol

        grad = image_space_comb - x0

        next_x = img - mu * grad   

        return next_x

class data_consistency():
    """
    Data consistency class can be used for:
        -performing E^h*E operation in the paper
        -transforming final network output to kspace
    """

    def __init__(self, sens_maps, mask):
        
        self.sens_maps = sens_maps
        self.mask = mask
        self.shape_list = mask.shape
        self.scalar = torch.sqrt(torch.tensor(self.shape_list[-2] * self.shape_list[-1], dtype=torch.float32))
        self.scalar = torch.complex(self.scalar, torch.zeros_like(self.scalar))

    def EhE_Op(self, img, mu):  # in paper eq.(6)
        """
        Performs (E^h*E+ mu*I) x
        """
        # E: operation of img -> coil_img -> masked-kspace coil_img  
        coil_imgs = self.sens_maps * img  # # batch x 16 x nrow x ncol
        kspace = torch_utils.torch_fftshift(torch.fft.fftn(torch_utils.torch_ifftshift(coil_imgs),dim=(-2, -1))) / self.scalar  
        masked_kspace = kspace * self.mask  # batch x 16 x nrow x ncol

        # Eh: operation of masked-kspace coil_img -> spatial-space coil_img * scalar -> times conj of sens_maps 
        image_space_coil_imgs = torch_utils.torch_ifftshift(torch.fft.ifftn(torch_utils.torch_fftshift(masked_kspace),dim=(-2, -1))) * self.scalar        
        image_space_comb = torch.sum(image_space_coil_imgs * torch.conj(self.sens_maps), dim=1)   # batch x nrow x ncol
        ispace = image_space_comb + mu * img

        return ispace

    def SSDU_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        and selects only loss mask locations(\Lambda) for computing loss
        """
        coil_imgs = self.sens_maps * img
        kspace = torch_utils.torch_fftshift(torch.fft.fftn(torch_utils.torch_ifftshift(coil_imgs),dim=(-2, -1))) / self.scalar
        masked_kspace = kspace * self.mask   # mask = batch x 1 x nrow x ncol

        return masked_kspace  # # batch x 16 x nrow x ncol

    def Supervised_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        """  
        coil_imgs = self.sens_maps * img
        kspace = torch_utils.torch_fftshift(torch.fft.fftn(torch_utils.torch_ifftshift(coil_imgs),dim=(-2, -1))) / self.scalar
        return kspace

def conj_grad(input_elems, mu_param):
    """
    Parameters
    ----------
    input_data : contains tuple of  reg output rhs = E^h*y + mu*z , sens_maps and mask
    rhs = nrow x ncol x 2  # in paper eq.(6)
    sens_maps : coil sensitivity maps ncoil x nrow x ncol
    mask : nrow x ncol
    mu : penalty parameter

    Encoder : Object instance for performing encoding matrix operations

    Returns
    -------
    data consistency output, nrow x ncol x 2

    """
    rhs, sens_maps, mask = input_elems 
    rhs = torch_utils.torch_real2complex(rhs)
   
    Encoder = data_consistency(sens_maps, mask)

    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = rhs.clone()
    rsold = torch.sum(torch.conj(r) * r).real

    i = 0    

    while i < 10 and rsold > 1e-10:
   
        Ap = Encoder.EhE_Op(p, mu_param)       
        alpha = rsold / torch.sum(torch.conj(p) * Ap).real
      
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(torch.conj(r) * r).real       
        beta = rsnew / rsold
        
        p = r + beta * p
        i += 1
        rsold = rsnew
    #print(x.shape)
    return torch_utils.torch_complex2real(x)

def dc_block(rhs, sens_maps, mask, mu):
    """
    DC block employs conjugate gradient for data consistency in PyTorch.
    """
    dc_block_output = conj_grad((rhs, sens_maps, mask), mu)

    return dc_block_output  # batch x nrow x ncol x 2

def SSDU_kspace_transform(nw_output, sens_maps, mask):
    """
    This function transforms unrolled network output to k-space at only unseen locations in training (\Lambda locations).
    Assumes data_consistency and its SSDU_kspace method can handle batched inputs.
    """
    nw_output = torch_utils.torch_real2complex(nw_output)  # batch x nrow x ncol

    # Assuming data_consistency can be initialized with batched sens_maps and mask,
    # and its method SSDU_kspace can process batched nw_output.
    Encoder = data_consistency(sens_maps, mask)
    nw_output_kspace = Encoder.SSDU_kspace(nw_output)

    return torch_utils.torch_complex2real(nw_output_kspace)

def SSDU_kspace_transform_gd(nw_output, sens_maps, mask):
    """
    This function transforms unrolled network output to k-space at only unseen locations in training (\Lambda locations).
    Assumes data_consistency and its SSDU_kspace method can handle batched inputs.
    """ 
    shape_list = mask.shape
    scalar = torch.sqrt(torch.tensor(shape_list[-2] * shape_list[-1], dtype=torch.float32))
    scalar = torch.complex(scalar, torch.zeros_like(scalar))
    
    img = nw_output
    coil_imgs = sens_maps * img
    kspace = torch_utils.torch_fftshift(torch.fft.fftn(torch_utils.torch_ifftshift(coil_imgs),dim=(-2, -1))) / scalar
    masked_kspace = kspace * mask   # mask = batch x 1 x nrow x ncol

    return masked_kspace  # # batch x 16 x nrow x ncol

def Supervised_kspace_transform(nw_output, sens_maps, mask):
    """
    This function transforms unrolled network output to k-space in PyTorch.
    """
    nw_output = torch_utils.torch_real2complex(nw_output)

    # Initialize an empty list to store k-space outputs
    kspace_outputs = []

    # Loop through the batch
    for i in range(nw_output.size(0)):
        nw_output_enc = nw_output[i]
        sens_maps_enc = sens_maps[i]
        mask_enc = mask[i]
        
        Encoder = data_consistency(sens_maps_enc, mask_enc)
        nw_output_kspace = Encoder.Supervised_kspace(nw_output_enc)

        kspace_outputs.append(nw_output_kspace)

    kspace = torch.stack(kspace_outputs)

    return torch_utils.torch_complex2real(kspace)

