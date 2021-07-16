"""
An implementation of a beta-VAE on Pytorch + Pytorch Lightning. 
Its optimization can be done either on a vanilla way or leveraging torch_constrained.

Inspired on: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""

from src.lit_constrained import LitConstrained
from src import nn_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from torchmetrics import MetricCollection

class BetaVAE(LitConstrained):
    def __init__(
        self, im_dim, im_channels, hid_dims, latent_dim, act_fun, act_args, 
        conv_channels, kernel_size, stride, pool_k_size=None, beta=None, constrained_kwargs={}):
        """ 
        Build a beta-VAE. The beta controls the relative importance of the KL
        divergence wrt the reconstruction loss in the objective function during 
        training.

        Args:
            im_dim (scalar int): fixed dimension of input images.
            im_channels (scalar int): number of image channels.
            hid_dims (int list): hidden dimensions of the MLP classifier. 
            latent_dim (scalar int): dimension of the latent space. 
            act_fun (torch.nn activation): class for the activation.
            act_args (list): arguments for the constructor of the activation.
            conv_channels (list, optional): channels for the convolutional 
                layers of the network. Defaults to None.
            kernel_size  (scalar int): kernel size of convolutional layers.
            stride  (scalar int): stride of convolutional layers.
            pool_k_size  (scalar int, optional): kernel size of max pooling layers.
                If not provided, no max pooling layer is included.
            beta (scalar, optional): Relative importance of the KL divergence in
                the VAE objective. This is overruled if the optimization is done via
                the constrained approach. 
            constrained_kwargs (dict, optional): arguments for the initialization
                of the LitConstrained class. Defaults to {}, where a single 
                objective optimization problem is considered during training.
        """
        super().__init__(**constrained_kwargs) 
        
        if self.is_constrained:
            print("If a beta value was provided, it will be ignored.")
        else: 
            self.beta = beta

        # ----------------------------------------- Encoder
        if conv_channels is not None:
            enc_conv_layers, bind_channels, bind_im_dim, residuals = nn_utils.build_convnet(
                im_channels, im_dim, conv_channels, kernel_size, stride, 
                pool_k_size, act_fun, act_args
                )
        else: 
            enc_conv_layers, bind_channels, bind_im_dim = [], im_channels, im_dim
        
        enc_conv_layers.append(nn.Flatten()) # flatten
        
        flat_bind_dim = bind_channels * bind_im_dim**2 # flat dim binding convnet with mlp net
        
        if hid_dims is not None:
            enc_mlp_layers = nn_utils.build_mlp(flat_bind_dim, hid_dims, act_fun, act_args)
            last_dim = hid_dims[-1]
        else:
            enc_mlp_layers = []
            last_dim = flat_bind_dim
        
        self.enc_network = nn.Sequential(*enc_conv_layers, *enc_mlp_layers)
        
        # Mu and sigma layers
        self.mu_layer = nn.Linear(last_dim,latent_dim)
        self.var_layer = nn.Linear(last_dim,latent_dim)
        
        # ----------------------------------------- Decoder
        if hid_dims is not None:
            dec_mlp_layers = nn_utils.build_mlp(latent_dim, hid_dims[::-1], act_fun, act_args)
            last_dim = hid_dims[0]
        else:
            dec_mlp_layers = []
            last_dim = latent_dim
        
        dec_mlp_layers.append(nn.Linear(last_dim, flat_bind_dim))
        dec_mlp_layers.append(nn.Unflatten(1, (bind_channels, bind_im_dim, bind_im_dim)))
        
        if conv_channels is not None:
            dec_conv_layers = nn_utils.build_rev_convnet(
                conv_channels[::-1], im_channels, kernel_size, stride, 
                pool_k_size, act_fun, act_args, residuals[::-1]
                )
            # Last layer
            dec_conv_layers.append(
                nn.Conv2d(conv_channels[0], im_channels, kernel_size, 1, "same")
                )
        else:
            dec_conv_layers = []
        
        self.decode = nn.Sequential(*dec_mlp_layers, *dec_conv_layers, nn.Sigmoid())
        
        # Set metrics - hardcoded.
        self.metrics = MetricCollection([
            tm.MeanSquaredError(),
        ])        
    
    def encode(self, sample):
        """
        Get the distribution parameters of encoded data derived from a sample. 

        Args:
            sample (n x ambient_dim tensor): a batch of data.

        Returns:
            n x latent_dim tensor: tensor of means.
            n x latent_dim tensor: tensor of log_variances. 
        """
        hid_state = self.enc_network(sample)
        return self.mu_layer(hid_state), self.var_layer(hid_state)
    
    def reparameterize(self, mu, logvar):
        """
        Reparametrization trick. Samples a datapoint from the normal distribution
        induced by mu and logvar. 

        Args:
            mu (n x latent_dim tensor): tensor of means.
            logvar (n x latent_dim tensor): tensor of log_variances. 

        Returns:
            n x latent_dim tensor: a datapoint on the latent space
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, sample):
        """
        Deduce a distribution on the latent space corresponding to possible encodings
        of sample. Then, sample a point from this distribution and decode it. 
        
        Args:
            sample (n x data_dim tensor): a batch of data.

        Returns:
            n x ambient_dim tensor: tensor of log_variances. 
            n x latent_dim tensor: tensor of means.
            n x latent_dim tensor: tensor of log_variances. 
        """
        mu, logvar = self.encode(sample)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def eval_losses(self, sample):
        """
        Evaluates the reconstruction loss and KL divergence of encoded representations
        on the current state of the VAE.

        Args:
            sample (n x data_dim tensor): a batch of data.

        Returns:
            tensor: either the reconstruction loss or the ELBO (which includes the KL divergence)
            Nonetype
            tensor: KL divergence of encoded representations.
        """
        x, _ = sample
        recon, mu, logvar = self(x) 
        loss = F.mse_loss(recon, x) # BCE is not symetrical
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        
        if self.is_constrained is False: 
            loss = loss + self.beta * KLD            
        
        return loss, None, KLD.view(1,1)
        
    def log_metrics(self, batch):
        """
        Log a selection of metrics.
        """
        x, _ = batch
        recon, _, _ = self(x) 
        metric_dict = self.metrics(recon, x)
        self.log_dict(metric_dict, on_epoch=True)