from torch.nn.modules.batchnorm import BatchNorm2d
from lit_constrained import LitConstrained
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from torchmetrics import MetricCollection

class BetaVAE(LitConstrained):
    def __init__(
        self, im_dim, im_channels, hid_dims, latent_dim, act_fun, act_args, 
        conv_channels, kernel_size, stride, pool_k_size, beta=None, constrained_kwargs={}):
        """ Mention beta is over-ruled by constrained

        Args:
            im_dim ([type]): [description]
            im_channels ([type]): [description]
            hid_dims ([type]): [description]
            latent_dim ([type]): [description]
            act_fun ([type]): [description]
            act_args ([type]): [description]
            conv_channels ([type]): [description]
            kernel_size ([type]): [description]
            stride ([type]): [description]
            pool_k_size ([type]): [description]
            beta ([type], optional): [description]. Defaults to None.
            constrained_kwargs (dict, optional): [description]. Defaults to {}.
        """
        super().__init__(**constrained_kwargs) 
        
        if self.is_constrained:
            print("If a beta value was provided, it will be ignored.")
        else: 
            self.beta = beta

        # ----------------------------------------- Encoder
        if conv_channels is not None:
            enc_conv_layers, bind_channels, bind_im_dim = utils.build_convnet(
                im_channels, im_dim, conv_channels, kernel_size, stride, 
                pool_k_size, act_fun, act_args
                )
        else: 
            enc_conv_layers, bind_channels, bind_im_dim = [], im_channels, im_dim
        
        flat_bind_dim = bind_channels * bind_im_dim**2 # flat dim binding convnet with mlp net
        enc_mlp_layers = utils.build_mlp(flat_bind_dim, hid_dims, act_fun, act_args)
        enc_layers = [*enc_conv_layers, *enc_mlp_layers]
        
        self.enc_network = nn.Sequential(*enc_layers)
        
        # Mu and sigma layers
        self.mu_layer = nn.Linear(hid_dims[-1],latent_dim)
        self.var_layer = nn.Linear(hid_dims[-1],latent_dim)
        
        # ----------------------------------------- Decoder
        dec_mlp_layers = utils.build_mlp(latent_dim, hid_dims[::-1], act_fun, act_args)
        dec_mlp_layers.append(nn.Linear(hid_dims[0], flat_bind_dim))
        dec_mlp_layers.append(nn.Unflatten(1, (bind_channels, bind_im_dim, bind_im_dim)))
        
        if conv_channels is not None:
            dec_conv_layers = utils.build_rev_convnet(
                conv_channels[::-1], im_channels, kernel_size, stride, 
                pool_k_size, act_fun, act_args
                )
        else:
            dec_conv_layers = []
        
        dec_layers = [*dec_mlp_layers, *dec_conv_layers, nn.Sigmoid()]
        self.decode = nn.Sequential(*dec_layers)
        
        print(self.encode)
        print(self.decode)
        
        # Set metrics - hardcoded. Check micro/macro flag
        self.metrics = MetricCollection([
            tm.MeanSquaredError(),
        ])        
    
    def encode(self, sample):
        hid_state = self.enc_network(sample)
        return self.mu_layer(hid_state), self.var_layer(hid_state)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, sample):
        mu, logvar = self.encode(sample)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def eval_losses(self, sample):
        x, _ = sample
        recon, mu, logvar = self(x) 
        BCE = F.binary_cross_entropy(recon, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        if self.is_constrained is False: 
            KLD = self.beta * KLD            
        
        return BCE, KLD.view(1,1), None
        
    def log_metrics(self, batch):
        x, _ = batch
        recon, _, _ = self(x) 
        metric_dict = self.metrics(recon, x)
        self.log_dict(metric_dict, on_epoch=True)