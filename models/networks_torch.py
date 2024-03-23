import torch
import torch.nn as nn
import torch.nn.functional as F

def res_block():
    return nn.Sequential(
        ConvLayer((64, 64, 3, 3), is_relu=True, is_scaling=False),
        ConvLayer((64, 64, 3, 3), is_relu=False, is_scaling=True)
    )

class ConvLayer(nn.Module):
    def __init__(self, conv_filter, is_relu=False, is_scaling=False):
        super(ConvLayer, self).__init__()
        
        in_channels, out_channels, filter_height, filter_width = conv_filter

        self.conv = nn.Conv2d(in_channels=in_channels, 
                            out_channels=out_channels, 
                            kernel_size=(filter_height, filter_width),
                            padding=1)
        
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        
        nn.init.normal_(self.conv.weight, mean=0, std=0.05)
        
        self.is_relu = is_relu
        self.is_scaling = is_scaling

    def forward(self, x):
        
        x = self.conv(x)
        # x = self.bn(x)
        
        if self.is_relu:
            x = F.relu(x)
                        
        if self.is_scaling:
            x = x * 0.1
        return x

class ResNet(nn.Module):
    """

    Parameters
    ----------
    input_data : nrow x ncol x 2 . Regularizer Input ---> change to ---> (batch_size, channels=2, nrow x ncol)
    nb_res_blocks : default is 15.

    conv_filters : dictionary containing size of the convolutional filters applied in the ResNet
    intermediate outputs : dictionary containing intermediate outputs of the ResNet

    Returns
    -------
    nw_output :  nrow x ncol x 2 . Regularizer output

    """
    def __init__(self, nb_res_blocks):
        super(ResNet, self).__init__()        
        layers = []
        self.nb_res_blocks = nb_res_blocks
        self.first_layer = ConvLayer((2, 128, 3, 3), is_relu=False, is_scaling=False)    

        for _ in range(self.nb_res_blocks):  
            layers += res_block()
            
        self.rbs = nn.Sequential(*layers)
        #print(len(self.rbs))
 
        self.second_last_layer = ConvLayer((128, 128, 3, 3), is_relu=False, is_scaling=False)
        self.last_layer = ConvLayer((128, 2, 3, 3), is_relu=False, is_scaling=False)
    
    def forward(self, x):
        # Assuming input_tensor with shape (batch, nrow, ncol, 2)
        x = x.permute(0, 3, 1, 2)
        xk = self.first_layer(x)       
        xii = xk.clone()
        
        for i, layer in enumerate(self.rbs):
            if (i % 2 == 0): 
                xi = xk.clone()
                #print(i)
            
            xk = layer(xk)
            
            #############
            if (i % 2 == 1): 
                xk = xk + xi
            #############

        xk = self.second_last_layer(xk)
        xk = xk + xii  # if this is a correct structure then it is not the same as fig.1 from paper
        nw_output = self.last_layer(xk)
        nw_output = nw_output.permute(0, 2, 3, 1)         
        return nw_output
    
class ResNet_modl(nn.Module):
    """

    Parameters
    ----------
    input_data : nrow x ncol x 2 . Regularizer Input ---> change to ---> (batch_size, channels=2, nrow x ncol)
    nb_res_blocks : default is 15.

    conv_filters : dictionary containing size of the convolutional filters applied in the ResNet
    intermediate outputs : dictionary containing intermediate outputs of the ResNet

    Returns
    -------
    nw_output :  nrow x ncol x 2 . Regularizer output

    """
    def __init__(self, nb_res_blocks):
        super(ResNet_modl, self).__init__()        
        layers = []
        self.nb_res_blocks = nb_res_blocks
        self.first_layer = ConvLayer((2, 64, 3, 3), is_relu=False, is_scaling=False)    

        for _ in range(self.nb_res_blocks):  
            layers += res_block()
            
        self.rbs = nn.Sequential(*layers)
        #print(len(self.rbs))
 
        self.second_last_layer = ConvLayer((64, 64, 3, 3), is_relu=False, is_scaling=False)
        self.last_layer = ConvLayer((64, 2, 3, 3), is_relu=False, is_scaling=False)
    
    def forward(self, x):
        # input_tensor with shape (batch, 2, nrow, ncol)
        xk = self.first_layer(x)       
        xii = xk.clone()
        
        for i, layer in enumerate(self.rbs):
            if (i % 2 == 0): 
                xi = xk.clone()
                #print(i)
            
            xk = layer(xk)
            
            #############
            if (i % 2 == 1): 
                xk = xk + xi
            #############

        xk = self.second_last_layer(xk)
        xk = xk + xii  # if this is a correct structure then it is not the same as fig.1 from paper
        nw_output = self.last_layer(xk)         
        return nw_output

