#Transformer denoiser ======================
'''
Patch Embedding:
The input image is divided into patches. Each patch is flattened and transformed into a vector 
using a linear projection (embedding). Positional encodings are added to these patch embeddings
to retain positional information.

Transformer Encoder:
The sequence of patch embeddings is passed through a series of Transformer encoder layers.
Each encoder layer uses self-attention to weigh the importance of other patches when encoding a
particular patch. This allows the model to consider the entire image context when processing each
patch, which is beneficial for distinguishing between noise and signal.

Transformer Decoder (if present):
In some transformer models, a decoder can be employed to generate the output sequence from the
encoded representations. The decoder may also use self-attention and cross-attention (attending
to encoder outputs) mechanisms to refine the feature representations.

Output Projection:
The output from the transformer encoder (or decoder) is then projected back to the image space.
This typically involves reshaping the output vectors into the shape of the image patches and then
reconstructing the full image from these patches. An additional convolutional layer or other types
of network heads can be applied to further refine the output and generate the clean image.

Training:
The transformer is trained on pairs of noisy and clean images.
The loss function used during training measures the difference between the transformer's output
and the clean target image, guiding the transformer to generate denoised outputs.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Transformer_full(nn.Module):
    def __init__(self, n_layers):
        super().__init__()        
        self.num_encoder_layers = n_layers        
        self.out_channels = 2
        self.embed_size = 384*2   # old: 384*2  The size of the embedding for each patch. 
        self.num_heads = 12       # old: 12
        
        self.num_patches = 2112  # (384 // patch_size) * (384 // patch_size) = 576     old: 576
        self.scale_factor = 12  # equal to patch_size = 16                            old: 16
        
        self.VisionTransformer = VisionTransformer(num_patches=self.num_patches, embed_size=self.embed_size, num_encoder_layers=self.num_encoder_layers, num_heads=self.num_heads) #.to(device)
        self.FeatureMapToImage = FeatureMapToImage2(in_channels=self.embed_size, out_channels=self.out_channels, scale_factor=self.scale_factor) #.to(device)
            
    def forward(self, x0):
        #x0 = x0.to(device)
        batch_size = x0.shape[0]
        image_size = (x0.shape[2], x0.shape[3])    
        patch_size = int(math.sqrt((x0.shape[2] * x0.shape[3]) / self.num_patches))
             
        patches = input_embeddings(x0, patch_size=patch_size, embed_size=self.embed_size, num_patches=self.num_patches)
        inter = self.VisionTransformer(patches)
        inter = change_shape(inter, image_size=image_size, patch_size=patch_size, batch_size=batch_size, embed_size=self.embed_size)
        out = self.FeatureMapToImage(inter)  
        return out      
        

# Add positional encodings
class PatchEmbedding(nn.Module):

    def __init__(self, num_patches, embed_size):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, embed_size), requires_grad=True)  #.to(device)
        # The position embeddings, with a length defined by embed_size, provide where each patch is located within the overall image.
    def forward(self, x): # x = (batch_size, num_patches, embed_size)
        x = x + self.position_embeddings
        return x

class VisionTransformer(nn.Module):
    def __init__(self, num_patches, embed_size, num_encoder_layers, num_heads):
        super().__init__()       
        ''' #### nn.TransformerEncoderLayer ###
        attention_output = self-attention(x) + x   # Self-attention and residual connection
        y = LayerNorm(attention_output)            # Normalization
        ffn_output = feed-forward(y) + y           # Feed-forward network and residual connection
        layer_output = LayerNorm(ffn_output)       # Normalization '''        
        self.patch_embedding = PatchEmbedding(num_patches, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # d_model defines the dimensionality of the input and output features for the encoder layer.
        # nhead should be a divisor of d_model. within each attention head,the d_model dimension is split evenly across all heads.

    def forward(self, x): # x = (batch_size, num_patches, embed_size)
        x = self.patch_embedding(x)  # Apply patch embedding and position encoding
        x = x.permute(1, 0, 2) 
        x = self.transformer_encoder(x)
        return x #(num_patches, batch_size, embed_size)

class FeatureMapToImage(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.upsample(x)
        return x

class FeatureMapToImage2(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        # Upsampling and convolution blocks
        layers = []
        num_upsample_blocks = int(np.floor(scale_factor / 2))
        for _ in range(num_upsample_blocks):
            
            inter_channels = max(1, in_channels // 2)  # Ensure out_channels is at least 1
            
            layers.extend([
                nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                            
            ])
            
            in_channels = max(1, inter_channels)  # Ensure in_channels for next layer is valid

        # Final convolution to get the desired number of output channels
        layers.append(nn.Conv2d(inter_channels, out_channels, kernel_size=1))
        layers.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



def change_shape(x, image_size, patch_size, batch_size, embed_size):
    # x is the output from the VisionTransformer (num_patches, batch_size, embed_size) 
    num_patches_h = image_size[0]//patch_size
    num_patches_w = image_size[1]//patch_size
     
    x = x.permute(1, 0, 2)  # Change to (batch_size, num_patches, embed_size)
    x = x.contiguous().view(batch_size, embed_size, num_patches_h, num_patches_w) 
    return x

# Create patches and embed them : input embedding
def input_embeddings(image, patch_size, embed_size, num_patches):
    assert image.size(2) % patch_size == 0 and image.size(3) % patch_size == 0, "Image dimensions must be divisible by the patch size."
    
    batch_size, channels, height, width = image.shape

    # Create patches
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # After these operations, patches is a 6D tensor with dimensions 
    # [batch_size, channels, num_patches_height, num_patches_width, patch_size, patch_size].
    patches = patches.contiguous().view(batch_size, channels, num_patches, -1)

    # Flatten patches and embed
    patches = patches.view(batch_size, num_patches, -1)
    patch_embedding_layer = nn.Linear(patch_size * patch_size * channels, embed_size)    
    patch_embedding_layer.to(image.device)
    patches = patch_embedding_layer(patches)
    # patches = (batch_size, num_patches, embed_size)
    return patches




'''
if __name__ == '__main__':
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.randint(low=0, high=10, size=(10, 2, 384, 384)).to(torch.float32)
    image = image.to(device)
    #print(image.shape) # ([10, 2, 384, 384])
    tr = Transformer_full(5)
    out = tr(image)
    #print(out.shape) # ([10, 2, 384, 384])
'''    
    
    


