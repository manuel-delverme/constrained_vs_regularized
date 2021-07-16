"""
This script contains some useful functions.
"""
from torch import nn

def build_mlp(d_in, hid_dims, act_fun, act_args):
    """
    Builds a general MLP block with batch normalization. 
    This ends with an activation function. Consider adding output layers afterwards.
    
    Args:
        d_in (scalar int): flat input dimensions of the MLP.
        hid_dims (list): hidden dimensions of the MLP.
        act_fun (torch.nn activation): class for the activation.
        act_args (list): arguments for the constructor of the activation.

    Returns:
        nn.Sequential: sequence of nn.Modules.
    """
    layers = []
    for d_out in hid_dims:
        layers = layers + [
            nn.Linear(d_in, d_out),
            nn.BatchNorm1d(d_out),
            act_fun(*act_args)
            ] 
        d_in = d_out        
    return layers

def build_convnet(in_channels, im_dim, conv_channels, kernel_size, stride, 
                act_fun, act_args, pool_k_size=None):
    """
    Builds a general Convolutional network with batch normalization and possibly
    max pooling for 2D inputs. 
    
    Returns auxiliary objects which allow to build a reversed network of the one 
    produced which preserves the layer by layer dimensionality of samples.  
    
    Args:
        in_channels (scalar int): input channels to the network.
        conv_channels (scalar int): number of filters per layer of the network.
        kernel_size  (scalar int): kernel size of convolutional layers.
        stride  (scalar int): stride of convolutional layers.
        pool_k_size  (scalar int, optional): kernel size of max pooling layers.
            If not provided, no max pooling layer is included.
        act_fun (torch.nn activation): class for the activation.
        act_args (list): arguments for the constructor of the activation.

    Returns:
        nn.Sequential: sequence of nn.Modules.
        scalar int: number of filters of the output layer of the network. 
        scalar int: the width and height dimensions of the network's output.
        residuals: the number of pixels lost after each layer of the network due
            to the stride and kernel_size parameters. 
    """
    current_dim = im_dim # number of W and H pixels at the start
    layers = [] # convolutional layers
    residuals = [] # number of pixels lost on each layer
    
    for out_channels in conv_channels:
        layers = layers + [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            act_fun(*act_args),
            ] 
        
        # Max pooling
        if pool_k_size is not None: layers.append(nn.MaxPool2d(pool_k_size))
        
        # keep track of the output channels, lost pixels at each layer
        current_dim, residuals = get_conv_dim(
            current_dim, kernel_size, stride, pool_k_size, residuals
            )
        
        in_channels = out_channels
    
    return layers, out_channels, current_dim, residuals

def build_rev_convnet(conv_channels, kernel_size, stride, 
                    pool_k_size, act_fun, act_args, residuals):
    """
    Builds a convnet with transposed convolutional layers. Pooling layers are
    replaced by upsampling layers. This is intended as a dimensional inverse of 
    build_convnet, given the same parameters. The residuals variable indicates
    the ammount of pixels lost to filtering and pooling layers of build_convnet.

    Args:
        conv_channels (scalar int): number of filters per layer of the network.
        kernel_size  (scalar int): kernel size of convolutional layers.
        stride  (scalar int): stride of convolutional layers.
        pool_k_size  (scalar int, optional): kernel size of max pooling layers.
            If not provided, no max pooling layer is included.
        act_fun (torch.nn activation): class for the activation.
        act_args (list): arguments for the constructor of the activation.

    Returns:
        nn.Sequential: sequence of nn.Modules.
    """
    layers = []
    in_channels = conv_channels[0]
    
    layer_n = 0
    for out_channels in conv_channels[1:]:
        # Upsampling/unpooling
        if pool_k_size is not None:
            layers = layers + [
                nn.Upsample(scale_factor=pool_k_size),
                nn.ReplicationPad2d((0,residuals[layer_n],0,residuals[layer_n])), 
                ]
            layer_n += 1
        
        # ConvTranspose
        layers = layers + [
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, output_padding=residuals[layer_n]
                ),
            nn.BatchNorm2d(out_channels),
            act_fun(*act_args),
            ] 
        layer_n += 1
        
        in_channels = out_channels
    
    # Last layer, without activation
    if pool_k_size is not None:
        layers = layers + [
            nn.Upsample(scale_factor=pool_k_size),
            nn.ReplicationPad2d((0,residuals[-2],0,residuals[-2])),
        ]
    layers.append(nn.ConvTranspose2d(
        conv_channels[-1], conv_channels[-1], kernel_size, stride, output_padding=residuals[-1]
        ))
    
    return layers

def get_conv_dim(in_dim, kernel_size, stride, pool_k_size, residuals):
    """
    Determines the output dimension of a block consisting of a convolutional 
    layer and a max pooling layer. Also, records the ammount of pixels lost on
    every layer due to the specific dimensions of the input. 
    
    Args:
        in_dim (scalar int): input dimension to the block.
        kernel_size  (scalar int): kernel size of the convolutional layer.
        stride  (scalar int): stride of the convolutional layer.
        pool_k_size  (scalar int): kernel size of the max pooling layer.

    Returns:
        scalar int: output dimension of the convolutional block.
    """
    residuals.append((in_dim - kernel_size) % stride)
    dim_after_conv = int((in_dim - kernel_size)/stride + 1)
    if pool_k_size is not None:
        residuals.append((dim_after_conv - pool_k_size) % pool_k_size)
        dim_after_conv = int((dim_after_conv - pool_k_size)/pool_k_size + 1)

    return dim_after_conv, residuals


