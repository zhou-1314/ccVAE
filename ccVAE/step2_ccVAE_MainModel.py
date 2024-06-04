from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F

from .step2_ccVAE_layers import Encoder, Decoder
from .step2_ccVAE_loss import mse, mmd, zinb, nb, center_loss
from .step2_ccVAE_utils import one_hot_encoder


class ccVAE_ccPairs(nn.Module):
    """ccVAE_ccPairs model class. This class contains the implementation of Transfer Variational Auto-encoder.

       Parameters
       ----------
       input_dim: Integer
            Number of input features (i.e. gene in case of scRNA-seq).
       conditions: List
            List of Condition names that the used data will contain to get the right encoding when used after reloading.
       hidden_layer_sizes: List
            A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
       latent_dim: Integer
            Bottleneck layer (z)  size.
       use_ccPairs: String
            Which type of cells are used for model construction, and must be one of the `ccPairs`, `rest` or `concat`.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`==0 no dropout will be applied.
       use_mmd: Boolean
            If 'True' an additional MMD loss will be calculated on the latent dim. 'z' or the first decoder layer 'y'.
       mmd_on: String
            Choose on which layer MMD loss will be calculated on if 'use_mmd=True': 'z' for latent dim or 'y' for first
            decoder layer.
       mmd_boundary: Integer or None
            Choose on how many conditions the MMD loss should be calculated on. If 'None' MMD will be calculated on all
            conditions.
       recon_loss: String
            Definition of Reconstruction-Loss-Method, 'mse', 'nb' or 'zinb'.
       beta: Float
            Scaling Factor for MMD loss. Higher beta values result in stronger batch-correction at a cost of worse biological variation.
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
    """

    def __init__(self,
                 input_dim: int,
                 conditions: list,
                 hidden_layer_sizes: list = [256, 64],
                 latent_dim: int = 10,
                 use_ccPairs: str = 'ccPairs',
                 dr_rate: float = 0.05,
                 use_mmd: bool = True,
                 mmd_on: str = 'z',
                 mmd_boundary: Optional[int] = None,
                 recon_loss: Optional[str] = 'nb',
                 beta: float = 1,
                 use_bn: bool = False,
                 use_ln: bool = True,
                 ):
        super().__init__()
        assert isinstance(hidden_layer_sizes, list)
        assert isinstance(latent_dim, int)
        assert isinstance(conditions, list)
        assert recon_loss in ["mse", "nb", "zinb"], "'recon_loss' must be 'mse', 'nb' or 'zinb'"

        print("\nInitializing ccVAE SubNetwork(trVAE) for ccPairs cells:")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_ccPairs = use_ccPairs
        self.n_conditions = len(conditions)
        self.conditions = conditions
        self.recon_loss = recon_loss
        self.mmd_boundary = mmd_boundary
        self.use_mmd = use_mmd
        self.beta = beta
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.mmd_on = mmd_on

        self.dr_rate = dr_rate
        if self.dr_rate > 0:
            self.use_dr = True
        else:
            self.use_dr = False

        if recon_loss in ["nb", "zinb"]:
            self.theta = torch.nn.Parameter(torch.randn(self.input_dim, self.n_conditions))
        else:
            self.theta = None

        self.hidden_layer_sizes = hidden_layer_sizes
        encoder_layer_sizes = self.hidden_layer_sizes.copy()
        encoder_layer_sizes.insert(0, self.input_dim)
        decoder_layer_sizes = self.hidden_layer_sizes.copy()
        decoder_layer_sizes.reverse()
        decoder_layer_sizes.append(self.input_dim)
        self.encoder = Encoder(encoder_layer_sizes,
                               self.latent_dim,
                               self.use_ccPairs,
                               self.use_bn,
                               self.use_ln,
                               self.use_dr,
                               self.dr_rate,
                               self.n_conditions)
        self.decoder = Decoder(decoder_layer_sizes,
                               self.latent_dim,
                               self.use_ccPairs,
                               self.recon_loss,
                               self.use_bn,
                               self.use_ln,
                               self.use_dr,
                               self.dr_rate,
                               self.n_conditions) 

    def sampling(self, mu, log_var):
        """Samples from standard Normal distribution and applies re-parametrization trick.
           It is actually sampling from latent space distributions with N(mu, var), computed by encoder.

           Parameters
           ----------
           mu: torch.Tensor
                Torch Tensor of Means.
           log_var: torch.Tensor
                Torch Tensor of log. variances.

           Returns
           -------
           Torch Tensor of sampled data.
        """
        var = torch.exp(log_var) + 1e-4
        return Normal(mu, var.sqrt()).rsample()


    def get_latent(self, x_ccPairs, batch_ccPairs=None, mean=False, mean_var=False):
        """
        Map input tensor `x_ccPairs` to the latent space.
    
        Parameters:
            x_ccPairs: torch.Tensor
                Input tensor to be mapped to the latent space. It should have shape [n_obs, input_dim].
    
            batch_ccPairs: torch.Tensor, optional
                Tensor of batch labels for each sample.
    
            mean: bool, optional (default: False)
                If True, return the mean of the latent space encoding.
    
            mean_var: bool, optional (default: False)
                If True, return both the mean and variance of the latent space encoding.
    
        Returns:
            If `mean` is True:
                Torch Tensor containing the mean of the latent space encoding of `x_ccPairs`.
    
            If `mean_var` is True:
                Tuple containing the mean and variance of the latent space encoding of `x_ccPairs`.
    
            Otherwise:
                Torch Tensor containing the latent space encoding of `x_ccPairs`.
        """
        x_ccPairs_ = torch.log(1 + x_ccPairs)

        if self.recon_loss == 'mse':
            x_ccPairs_ = x_ccPairs

        z_ccPairs_mean, z_ccPairs_log_var = self.encoder(x_ccPairs=x_ccPairs_, batch_ccPairs=batch_ccPairs)
        latent_ccPairs = self.sampling(z_ccPairs_mean, z_ccPairs_log_var)
        if mean:
            return z_ccPairs_mean
        elif mean_var:
            return (z_ccPairs_mean, torch.exp(z_ccPairs_log_var) + 1e-4)
        return latent_ccPairs



    def get_y(self, x_ccPairs, batch_ccPairs=None, use_decoder='ccPairs'):
        """
        Map input tensor `x_ccPairs` to the y dimension (first layer of decoder).
    
        Parameters:
            x_ccPairs: torch.Tensor
                Input tensor to be mapped to the latent space. It should have shape [n_obs, input_dim].
    
            batch_ccPairs: torch.Tensor, optional
                Tensor of batch labels for each sample.
    
            use_decoder: str, optional (default: 'ccPairs')
                Specifies which decoder to use.
    
        Returns:
            Torch Tensor containing the output of the first decoder layer.
        """
        x_ccPairs_ = torch.log(1 + x_ccPairs)

        if self.recon_loss == 'mse':
            x_ccPairs_ = x_ccPairs

        z_ccPairs_mean, z_ccPairs_log_var = self.encoder(x_ccPairs=x_ccPairs_, batch_ccPairs=batch_ccPairs)
        z1_ccPairs = self.sampling(z_ccPairs_mean, z_ccPairs_log_var)
        output_ccPairs = self.decoder(x_ccPairs=z1_ccPairs, batch_ccPairs=batch_ccPairs, use_decoder=use_decoder)
        return output_ccPairs[-1]
        
        

class ccVAE_rest(nn.Module):
    """ScArches model class. This class contains the implementation of Conditional Variational Auto-encoder.

       Parameters
       ----------
       input_dim: Integer
            Number of input features (i.e. gene in case of scRNA-seq).
       conditions: List
            List of Condition names that the used data will contain to get the right encoding when used after reloading.
       hidden_layer_sizes: List
            A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
       latent_dim: Integer
            Bottleneck layer (z)  size.
       use_ccPairs: String
            Which type of cells are used for model construction, and must be one of the `ccPairs`, `rest` or `concat`.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`==0 no dropout will be applied.
       recon_loss: String
            Definition of Reconstruction-Loss-Method, 'mse', 'nb' or 'zinb'.
       beta: Float
            Scaling Factor for MMD loss. Higher beta values result in stronger batch-correction at a cost of worse biological variation.
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
    """

    def __init__(self,
                 input_dim: int,
                 conditions: list,
                 hidden_layer_sizes: list = [256, 64],
                 latent_dim: int = 10,
                 use_ccPairs: str = 'rest',
                 dr_rate: float = 0.05,
                 recon_loss: Optional[str] = 'nb',
                 beta: float = 1,
                 use_bn: bool = False,
                 use_ln: bool = True,
                 ):
        super().__init__()
        assert isinstance(hidden_layer_sizes, list)
        assert isinstance(latent_dim, int)
        assert isinstance(conditions, list)
        assert recon_loss in ["mse", "nb", "zinb"], "'recon_loss' must be 'mse', 'nb' or 'zinb'"

        print("\nInitializing ccVAE SubNetwork(CVAE) for batch-free cells:")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_ccPairs = use_ccPairs
        self.n_conditions = len(conditions)
        self.conditions = conditions
        self.recon_loss = recon_loss
        self.beta = beta
        self.use_bn = use_bn
        self.use_ln = use_ln

        self.dr_rate = dr_rate
        if self.dr_rate > 0:
            self.use_dr = True
        else:
            self.use_dr = False

        if recon_loss in ["nb", "zinb"]:
            self.theta = torch.nn.Parameter(torch.randn(self.input_dim, self.n_conditions))
        else:
            self.theta = None

        self.hidden_layer_sizes = hidden_layer_sizes
        encoder_layer_sizes = self.hidden_layer_sizes.copy()
        encoder_layer_sizes.insert(0, self.input_dim)
        decoder_layer_sizes = self.hidden_layer_sizes.copy()
        decoder_layer_sizes.reverse()
        decoder_layer_sizes.append(self.input_dim)
        self.encoder = Encoder(encoder_layer_sizes,
                               self.latent_dim,
                               self.use_ccPairs,
                               self.use_bn,
                               self.use_ln,
                               self.use_dr,
                               self.dr_rate,
                               self.n_conditions)
        self.decoder = Decoder(decoder_layer_sizes,
                               self.latent_dim,
                               self.use_ccPairs,
                               self.recon_loss,
                               self.use_bn,
                               self.use_ln,
                               self.use_dr,
                               self.dr_rate,
                               self.n_conditions)

    def sampling(self, mu, log_var):
        """Samples from standard Normal distribution and applies re-parametrization trick.
           It is actually sampling from latent space distributions with N(mu, var), computed by encoder.

           Parameters
           ----------
           mu: torch.Tensor
                Torch Tensor of Means.
           log_var: torch.Tensor
                Torch Tensor of log. variances.

           Returns
           -------
           Torch Tensor of sampled data.
        """
        var = torch.exp(log_var) + 1e-4
        return Normal(mu, var.sqrt()).rsample()


    def get_latent(self, x_rest, batch_rest=None, mean=False, mean_var=False):
        """
        Map input tensor `x_rest` to the latent space.
    
        Parameters:
            x_rest: torch.Tensor
                Input tensor to be mapped to the latent space. It should have shape [n_obs, input_dim].
    
            batch_rest: torch.Tensor, optional
                Tensor of batch labels for each sample.
    
            mean: bool, optional (default: False)
                If True, return the mean of the latent space encoding.
    
            mean_var: bool, optional (default: False)
                If True, return both the mean and variance of the latent space encoding.
    
        Returns:
            If `mean` is True:
                Torch Tensor containing the mean of the latent space encoding of `x_rest`.
    
            If `mean_var` is True:
                Tuple containing the mean and variance of the latent space encoding of `x_rest`.
    
            Otherwise:
                Torch Tensor containing the latent space encoding of `x_rest`.
        """
        x_rest_ = torch.log(1 + x_rest)

        if self.recon_loss == 'mse':
            x_rest_ = x_rest

        z_rest_mean, z_rest_log_var = self.encoder(x_rest=x_rest_, batch_rest=batch_rest)
        latent_rest = self.sampling(z_rest_mean, z_rest_log_var)

        if mean:
            return z_rest_mean
        elif mean_var:
            return ( z_rest_mean, torch.exp(z_rest_log_var) + 1e-4)
        return latent_rest


    def get_y(self, x_rest, batch_rest=None, use_decoder='rest'):
        """
        Map input tensor `x_rest` to the y dimension (first layer of decoder).
    
        Parameters:
            x_rest: torch.Tensor
                Input tensor to be mapped to the latent space. It should have shape [n_obs, input_dim].
    
            batch_rest: torch.Tensor, optional
                Tensor of batch labels for each sample.
    
            use_decoder: str, optional (default: 'rest')
                Specifies which decoder to use.
    
        Returns:
            Torch Tensor containing the output of the first decoder layer.
        """
        x_rest_ = torch.log(1 + x_rest)
        if self.recon_loss == 'mse':
            x_rest_ = x_rest
        z_mean_rest, z_log_var_rest = self.encoder(x_rest=x_rest_, batch_rest=batch_rest)
        latent_rest = self.sampling(z_mean_rest, z_log_var_rest)
        output_rest = self.decoder(x_rest=latent_rest, batch_rest=batch_rest, use_decoder=use_decoder)
        return output_rest[-1]
      
      
class ccVAE(nn.Module):
    """Model for ccVAE class. This class contains the implementation of Conditional Variational Auto-encoder.

       Parameters
       ----------
        input_dim1 (int): 
          Input dimension of ccPairs data.
        input_dim2 (int): 
          Input dimension of rest data.
        condition_key (str, optional): 
          Column name of conditions in `adata.obs` data frame. Defaults to None.
        conditions (list, optional): 
          List of condition names that the used data will contain to get the right encoding when used after reloading. Defaults to None.
        cell_type_encoder_list (list, optional): 
          List of cell type names. Defaults to None.
        number_of_class (int, optional):
          Number of cell types. Defaults to None.
        hidden_layer_sizes (list, optional): 
          A list of hidden layer sizes for the encoder network. The decoder network will be in the reversed order. Defaults to [256, 64].
        latent_dim (int, optional): 
          Bottleneck layer (z) size. Defaults to 10.
        use_ccPairs (str, optional): 
          Specifies the use of ccPairs data. Defaults to 'ccPairs'.
        dr_rate (float, optional): 
          Dropout rate applied to all layers. If `dr_rate` == 0, no dropout will be applied. Defaults to 0.05.
        use_mmd (bool, optional): 
          If True, an additional Maximum Mean Discrepancy (MMD) loss will be calculated on the latent dim (z) or the first decoder layer (y). Defaults to True.
        mmd_on (str, optional): 
          Choose on which layer MMD loss will be calculated if `use_mmd=True`: 'z' for the latent dim or 'y' for the first decoder layer. Defaults to 'z'.
        mmd_boundary (int or None, optional): Choose on how many conditions the MMD loss should be calculated on. If None, MMD will be calculated on all conditions. Defaults to None.
        recon_loss (str or None, optional): 
          Definition of the Reconstruction-Loss-Method: 'mse', 'nb', or 'zinb'. Defaults to 'nb'.
        beta (float, optional): Scaling Factor for the MMD loss. Defaults to 1.
        use_bn (bool, optional): 
          If True, batch normalization will be applied to layers. Defaults to False.
        use_ln (bool, optional): 
          If True, layer normalization will be applied to layers. Defaults to True.
    
    """

    def __init__(self,
                 input_dim1: int,
                 input_dim2: int,
                 condition_key: str = None,
                 conditions: Optional[list] = None,
                 cell_type_encoder_list: Optional[list] = None,
                 number_of_class: int = None,
                 hidden_layer_sizes: list = [256, 64],
                 latent_dim: int = 10,
                 use_ccPairs: str = 'ccPairs',
                 dr_rate: float = 0.05,
                 use_mmd: bool = True,
                 mmd_on: str = 'z',
                 mmd_boundary: Optional[int] = None,
                 recon_loss: Optional[str] = 'nb',
                 beta: float = 1,
                 use_bn: bool = False,
                 use_ln: bool = True):
        super().__init__()

        self.condition_key = condition_key

        if conditions is None:
            if condition_key is not None:
                self.conditions = adata_ccPairs.obs[condition_key].unique().tolist()
            else:
                self.conditions = []
        else:
            self.conditions = conditions

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_conditions = len(conditions)
        self.condition_encoder = {k: v for k, v in zip(conditions, range(len(conditions)))}
        self.cell_type_encoder = cell_type_encoder_list
        self.number_of_class = number_of_class
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.latent_dim = latent_dim
        self.use_ccPairs = use_ccPairs
        self.dr_rate = dr_rate
        self.use_mmd = use_mmd
        self.freeze = False
        self.mmd_on = mmd_on
        self.mmd_boundary = mmd_boundary
        self.recon_loss = recon_loss
        self.beta = beta
        self.use_bn = use_bn
        self.use_ln = use_ln

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2

        if recon_loss == None:
            self.recon_loss = ["mse"] * 2
        elif len(recon_loss) == 2:
            self.recon_loss = recon_loss
        else:
            raise ValueError(
                "The length of losses must be 2."
            )
      
        self.theta = None
        input_dims = [self.input_dim1, self.input_dim2]
        for i, loss in enumerate(recon_loss):
            if loss in ["nb", "zinb"]:
                self.theta = torch.nn.Parameter(torch.randn(input_dims[i], self.n_conditions))
                break
        
        print('\033[1m' + '\nINITIALIZING THE WHOLE CCVAE NETWORK..............' + '\033[0m')

        # 实例化两个 VAE 模型
        self.vae1 = ccVAE_ccPairs(
            self.input_dim1,
            self.conditions,
            self.hidden_layer_sizes,
            self.latent_dim,
            self.use_ccPairs,
            self.dr_rate,
            self.use_mmd,
            self.mmd_on,
            self.mmd_boundary,
            self.recon_loss[0],
            self.beta,
            self.use_bn,
            self.use_ln,
        )
        self.vae2 = ccVAE_rest(
            self.input_dim2,
            self.conditions,
            self.hidden_layer_sizes,
            self.latent_dim,
            'rest', # self.use_ccPairs_ = 'rest'
            self.dr_rate,
            self.recon_loss[1],
            self.beta,
            self.use_bn,
            self.use_ln,
        )
    
    def sampling(self, mu, log_var):
        """Samples from standard Normal distribution and applies re-parametrization trick.
           It is actually sampling from latent space distributions with N(mu, var), computed by encoder.

           Parameters
           ----------
           mu: torch.Tensor
                Torch Tensor of Means.
           log_var: torch.Tensor
                Torch Tensor of log. variances.

           Returns
           -------
           Torch Tensor of sampled data.
        """
        var = torch.exp(log_var) + 1e-4
        return Normal(mu, var.sqrt()).rsample()
    
    def merge_latents(self, x_ccPairs, x_rest, batch_ccPairs=None, batch_rest=None, mean=False, mean_var=False):
        """Merge latent variables from ccPairs and rest data using a CVAE model.

        Parameters:
            cvae (ccVAE): The CVAE model.
            x_ccPairs (torch.Tensor): ccPairs data to be mapped to latent space.
            x_rest (torch.Tensor): rest data to be mapped to latent space.
            batch_ccPairs (torch.Tensor): Condition labels for ccPairs data.
            batch_rest (torch.Tensor): Condition labels for rest data.

        Returns:
            merged_latent (torch.Tensor): Merged latent variables.
        """
        # Get latent variables from ccPairs data
        latent_ccPairs = self.vae1.get_latent(x_ccPairs, batch_ccPairs, mean, mean_var)

        # Get latent variables from rest data
        latent_rest = self.vae2.get_latent(x_rest, batch_rest, mean, mean_var)

        # Merge latent variables
        merged_latent = torch.cat((latent_ccPairs, latent_rest), dim=0) ## 沿着行合并，增加样本量

        return merged_latent
      

    def get_y(self, x_ccPairs, x_rest, batch_ccPairs=None, batch_rest=None):
        """Map `x` to the y dimension (first layer of the decoder). This function feeds data into the encoder and returns
        the output y for each sample in the data.
    
        Parameters:
            x_ccPairs (torch.Tensor): 
              Torch tensor to be mapped to latent space. Shape: [n_obs_ccPairs, input_dim_ccPairs].
            x_rest (torch.Tensor): 
              Torch tensor to be mapped to latent space. Shape: [n_obs_rest, input_dim_rest].
            batch_ccPairs (torch.Tensor, optional): 
              Torch tensor of batch labels for ccPairs data. Shape: [n_obs_ccPairs].
                Defaults to None.
            batch_rest (torch.Tensor, optional): 
              Torch tensor of batch labels for rest data. Shape: [n_obs_rest].
                Defaults to None.
    
        Returns:
            torch.Tensor: Tensor containing the output of the first decoder layer. Shape: [n_obs_ccPairs + n_obs_rest, y_dim].
        """
        # Get latent variables from ccPairs data
        latent_ccPairs = self.vae1.get_latent(x_ccPairs, batch_ccPairs)

        # Get latent variables from rest data
        latent_rest = self.vae2.get_latent(x_rest, batch_rest)
        
        output_ccPairs = self.vae1.decoder(latent_ccPairs, batch_ccPairs=batch_ccPairs, use_decoder='ccPairs')
        output_rest = self.vae2.decoder(latent_rest, batch_rest=batch_rest, use_decoder='rest')
        return torch.cat((output_ccPairs[-1], output_rest[-1]), dim=0) ## 沿着行合并，增加样本量 
      
      
    def calc_recon_loss(self, xs, rs, losses, batch_ccPairs, batch_rest, sizefactor_ccPairs, sizefactor_rest, masks):
        loss = []
        for i, (r, loss_type) in enumerate(zip(rs, losses)):
            if len(r) != 2 and len(r.shape) == 3:
                r = r.squeeze()
            if loss_type == "mse":
                recon_x_concat, y1_concat = r
                recon_loss_concat = mse(recon_x_concat, xs).sum(dim=-1)
                loss.append(recon_loss_concat)
            elif loss_type == "nb":
                dec_mean_gamma_concat, y1_concat = r
                sizefactor_concat = torch.concat((sizefactor_ccPairs.unsqueeze(1),sizefactor_rest.unsqueeze(1)), dim = 0)
                size_factor_view_concat = sizefactor_concat.expand(dec_mean_gamma_concat.size(0), dec_mean_gamma_concat.size(1))
                dec_mean_concat = dec_mean_gamma_concat * size_factor_view_concat
                self.theta =  self.theta.to(self.device)
                dispersion_concat = F.linear(one_hot_encoder(torch.concat((batch_ccPairs,batch_rest),dim=0), self.n_conditions), self.theta)
                dispersion_concat = torch.exp(dispersion_concat)
                recon_loss_concat = -nb(x=xs, mu=dec_mean_concat, theta=dispersion_concat).sum(dim=-1)
                loss.append(recon_loss_concat)
            elif loss_type == "zinb":
                dec_mean_gamma_concat, dec_dropout_concat, y1_concat = r
                sizefactor_concat = torch.concat((sizefactor_ccPairs.unsqueeze(1),sizefactor_rest.unsqueeze(1)), dim = 0)
                size_factor_view_concat = sizefactor_concat.expand(dec_mean_gamma_concat.size(0), dec_mean_gamma_concat.size(1))
                dec_mean_concat = dec_mean_gamma_concat * size_factor_view_concat
                self.theta =  self.theta.to(self.device)
                dispersion_concat = F.linear(one_hot_encoder(torch.concat((batch_ccPairs,batch_rest),dim=0), self.n_conditions), self.theta)
                dispersion_concat = torch.exp(dispersion_concat)
                recon_loss_concat = -zinb(x=xs, mu=dec_mean_concat, theta=dispersion_concat, pi=dec_dropout_concat).sum(dim=-1)
                loss.append(recon_loss_concat)

        return (
            torch.sum(torch.stack(loss, dim=-1) * torch.stack(masks, dim=-1), dim=1),
            torch.mean(torch.stack(loss, dim=-1) * torch.stack(masks, dim=-1), dim=0), # calculate mean value
        )
        
    
    def calc_center_loss(self, x_ccPairs, x_rest, batch_ccPairs, batch_rest, label, masks):
        loss = []
        for i in range(2):
            embeddings = self.merge_latents(x_ccPairs, x_rest, batch_ccPairs, batch_rest) # shape: [batch size, latent_concat]
            Center_Loss = center_loss(embeddings, label) # self.center_weight = 1
            loss.append(Center_Loss)
        return (
                torch.sum(torch.stack(loss, dim=-1) * torch.stack(masks, dim=-1), dim=1),
                torch.mean(torch.stack(loss, dim=-1) * torch.stack(masks, dim=-1), dim=0), # calculate mean value
            )

      
    def forward(self, x_ccPairs=None, x_rest=None, batch_ccPairs=None, batch_rest=None, 
                sizefactor_ccPairs=None, labeled_ccPairs=None, sizefactor_rest=None, labeled_rest=None):
        if x_ccPairs is not None:
            x_ccPairs_log = torch.log(1 + x_ccPairs)
        if x_rest is not None:
            x_rest_log = torch.log(1 + x_rest)
            
        ## concat two adata
        x_concat_log = torch.concat((x_ccPairs_log, x_rest_log), dim=0)
        x_concat = torch.concat((x_ccPairs, x_rest), dim=0)
                    
        if self.recon_loss[0] == 'mse':
            x_ccPairs_log = x_ccPairs
            x_rest_log = x_rest
            ## concat two adata
            x_concat_log = torch.concat((x_ccPairs_log, x_rest_log), dim=0)
            x_concat = torch.concat((x_ccPairs, x_rest), dim=0)
        
        # z = self.merge_latents(x_ccPairs_log, x_rest_log, batch_ccPairs, batch_rest)
        z1_ccPairs_mean, z1_ccPairs_log_var = self.vae1.encoder(x_ccPairs=x_ccPairs_log, batch_ccPairs=batch_ccPairs)
        z1_rest_mean, z1_rest_log_var = self.vae2.encoder(x_rest=x_rest_log, batch_rest=batch_rest)
        
        # Get latent variables from ccPairs data
        z_ccPairs = self.sampling(z1_ccPairs_mean, z1_ccPairs_log_var)

        # Get latent variables from rest data
        z_rest = self.sampling(z1_rest_mean, z1_rest_log_var)
        
        ## concat 
        z_concat = torch.concat((z_ccPairs, z_rest), dim=0) ## 沿着行合并，增加样本量 

        ## repeat 
        z = z_concat.unsqueeze(1).repeat(1, 2, 1)
        zs = torch.split(z, 1, dim=1)
        
        ## output of concat 
        outputs_concats = [self.vae1.decoder(z.squeeze(1), batch_ccPairs=batch_ccPairs, batch_rest=batch_rest, use_decoder='concat') for z in zs]

        xs = torch.split(
            x_concat, [x_ccPairs_log.shape[0], x_rest_log.shape[0]], dim=0
        )  # list of tensors of len = n_mod, each tensor is of shape batch_size x mod_input_dim
        mask_list = [x.sum(dim=1) > 0 for x in xs]  # [batch_size] * num_modalities

        masks = []
        for i, mask in enumerate(mask_list):
            # 获取另一个元素的索引
            other_index = (i + 1) % len(mask_list)
            # 获取另一个元素的形状
            other_shape = mask_list[other_index].shape
            # 创建填充的张量
            padding_tensor = torch.zeros(other_shape, dtype=torch.bool, device=self.device)
            # 将填充的张量和原始张量按行维度拼接
            if i == 0:
                padded_tensor = torch.cat([mask, padding_tensor], dim=0)
            elif i ==1:
                padded_tensor = torch.cat([padding_tensor, mask], dim=0)
            # 将拼接后的张量添加到结果列表
            masks.append(padded_tensor)
        
        recon_loss_concat, modality_recon_losses = self.calc_recon_loss(
            x_concat_log, outputs_concats, self.recon_loss, batch_ccPairs, batch_rest, sizefactor_ccPairs, sizefactor_rest, masks
        )

        z1_concat_log_var = torch.concat((z1_ccPairs_log_var,z1_rest_log_var),dim=0)
        z1_concat_mean  = torch.concat((z1_ccPairs_mean,z1_rest_mean),dim=0)
        z1_var_concat = torch.exp(z1_concat_log_var) + 1e-4
        kl_div_concat = kl_divergence(
            Normal(z1_concat_mean, torch.sqrt(z1_var_concat)),
            Normal(torch.zeros_like(z1_concat_mean), torch.ones_like(z1_var_concat))
        ).sum(dim=1).mean()
        
        ## cluster loss
        if self.cell_type_encoder is not None:
            center_loss_concat, modality_center_loss = self.calc_center_loss(x_ccPairs_log, x_rest_log, batch_ccPairs, batch_rest, self.cell_type_encoder, masks)
            
        mmd_loss = torch.tensor(0.0, device=z_concat.device)

        if self.use_mmd:
            #batch_concat = torch.concat((batch_ccPairs,batch_rest),dim=0)
            if self.mmd_on == "z":
                mmd_loss = mmd(z_ccPairs, batch_ccPairs,self.n_conditions, self.beta, self.mmd_boundary)
            else:
                mmd_loss = mmd(y1_ccPairs, batch_ccPairs,self.n_conditions, self.beta, self.mmd_boundary)
                
        if self.cell_type_encoder is not None:
            return recon_loss_concat.mean(), kl_div_concat, mmd_loss, modality_recon_losses, center_loss_concat.mean(), modality_center_loss
        else:
            return recon_loss_concat.mean(), kl_div_concat, mmd_loss, modality_recon_losses
      
    
    # def forward(self, x_ccPairs=None, x_rest=None, batch_ccPairs=None, batch_rest=None, 
    #             sizefactor_ccPairs=None, labeled_ccPairs=None, sizefactor_rest=None, labeled_rest=None):
    #     if x_ccPairs is not None:
    #         x_ccPairs_log = torch.log(1 + x_ccPairs)
    #     if x_rest is not None:
    #         x_rest_log = torch.log(1 + x_rest)
    #         
    #     ## concat two adata
    #     x_concat_log = torch.concat((x_ccPairs_log, x_rest_log), dim=0)
    #     x_concat = torch.concat((x_ccPairs, x_rest), dim=0)
    #                 
    #     if self.recon_loss == 'mse':
    #         x_ccPairs_log = x_ccPairs
    #         x_rest_log = x_rest
    #         ## concat two adata
    #         x_concat_log = torch.concat((x_ccPairs_log, x_rest_log), dim=0)
    #         x_concat = torch.concat((x_ccPairs, x_rest), dim=0)
    #     
    #     # z = self.merge_latents(x_ccPairs_log, x_rest_log, batch_ccPairs, batch_rest)
    #     z1_ccPairs_mean, z1_ccPairs_log_var = self.vae1.encoder(x_ccPairs=x_ccPairs_log, batch_ccPairs=batch_ccPairs)
    #     z1_rest_mean, z1_rest_log_var = self.vae2.encoder(x_rest=x_rest_log, batch_rest=batch_rest)
    #     
    #     # Get latent variables from ccPairs data
    #     z_ccPairs = self.sampling(z1_ccPairs_mean, z1_ccPairs_log_var)
    # 
    #     # Get latent variables from rest data
    #     z_rest = self.sampling(z1_rest_mean, z1_rest_log_var)
    #     
    #     ## concat 
    #     z_concat = torch.concat((z_ccPairs, z_rest), dim=0) ## 沿着行合并，增加样本量 
    #     
    #     ## output of concat 
    #     outputs_concat = self.vae1.decoder(z_concat, batch_ccPairs=batch_ccPairs, batch_rest=batch_rest, use_decoder='concat')
    #     if self.recon_loss == "mse":
    #         recon_x_concat, y1_concat = outputs_concat
    #         recon_loss_concat = mse(recon_x_concat, x_concat_log).sum(dim=-1).mean()
    #         
    #     elif self.recon_loss == "zinb":
    #         dec_mean_gamma_concat, dec_dropout_concat, y1_concat = outputs_concat
    #         sizefactor_concat = torch.concat((sizefactor_ccPairs.unsqueeze(1),sizefactor_rest.unsqueeze(1)), dim = 0)
    #         size_factor_view_concat = sizefactor_concat.expand(dec_mean_gamma_concat.size(0), dec_mean_gamma_concat.size(1))
    #         dec_mean_concat = dec_mean_gamma_concat * size_factor_view_concat
    #         self.theta =  self.theta.to(self.device)
    #         dispersion_concat = F.linear(one_hot_encoder(torch.concat((batch_ccPairs,batch_rest),dim=0), self.n_conditions), self.theta)
    #         dispersion_concat = torch.exp(dispersion_concat)
    #         recon_loss_concat = -zinb(x=x_concat, mu=dec_mean_concat, theta=dispersion_concat, pi=dec_dropout_concat).sum(dim=-1).mean()
    #         
    #     elif self.recon_loss == "nb":
    #         dec_mean_gamma_concat, y1_concat = outputs_concat
    #         sizefactor_concat = torch.concat((sizefactor_ccPairs.unsqueeze(1),sizefactor_rest.unsqueeze(1)), dim = 0)
    #         size_factor_view_concat = sizefactor_concat.expand(dec_mean_gamma_concat.size(0), dec_mean_gamma_concat.size(1))
    #         dec_mean_concat = dec_mean_gamma_concat * size_factor_view_concat
    #         self.theta =  self.theta.to(self.device)
    #         dispersion_concat = F.linear(one_hot_encoder(torch.concat((batch_ccPairs,batch_rest),dim=0), self.n_conditions), self.theta)
    #         dispersion_concat = torch.exp(dispersion_concat)
    #         recon_loss_concat = -nb(x=x_concat, mu=dec_mean_concat, theta=dispersion_concat).sum(dim=-1).mean()
    # 
    #     z1_concat_log_var = torch.concat((z1_ccPairs_log_var,z1_rest_log_var),dim=0)
    #     z1_concat_mean  = torch.concat((z1_ccPairs_mean,z1_rest_mean),dim=0)
    #     z1_var_concat = torch.exp(z1_concat_log_var) + 1e-4
    #     kl_div_concat = kl_divergence(
    #         Normal(z1_concat_mean, torch.sqrt(z1_var_concat)),
    #         Normal(torch.zeros_like(z1_concat_mean), torch.ones_like(z1_var_concat))
    #     ).sum(dim=1).mean()
    # 
    #     ## outputs of ccPairs
    #     outputs_ccPairs = self.vae1.decoder(z_ccPairs, batch_ccPairs=batch_ccPairs, use_decoder='ccPairs')
    #     if self.recon_loss == "mse":
    #         recon_x_ccPairs, y1_ccPairs = outputs_ccPairs
    #         recon_loss_ccPairs = mse(recon_x_ccPairs, x_ccPairs_log).sum(dim=-1).mean()
    #         
    #     elif self.recon_loss == "zinb":
    #         dec_mean_gamma_ccPairs, dec_dropout_ccPairs, y1_ccPairs = outputs_ccPairs
    #         size_factor_view_ccPairs = sizefactor_ccPairs.unsqueeze(1).expand(dec_mean_gamma_ccPairs.size(0), dec_mean_gamma_ccPairs.size(1))
    #         dec_mean_ccPairs = dec_mean_gamma_ccPairs * size_factor_view_ccPairs
    #         self.theta =  self.theta.to(self.device)
    #         dispersion_ccPairs = F.linear(one_hot_encoder(batch_ccPairs, self.n_conditions), self.theta)
    #         dispersion_ccPairs = torch.exp(dispersion_ccPairs)
    #         recon_loss_ccPairs = -zinb(x=x_ccPairs, mu=dec_mean_ccPairs, theta=dispersion_ccPairs, pi=dec_dropout_ccPairs).sum(dim=-1).mean()
    #         
    #     elif self.recon_loss == "nb":
    #         dec_mean_gamma_ccPairs, y1_ccPairs = outputs_ccPairs
    #         size_factor_view_ccPairs = sizefactor_ccPairs.unsqueeze(1).expand(dec_mean_gamma_ccPairs.size(0), dec_mean_gamma_ccPairs.size(1))
    #         dec_mean_ccPairs = dec_mean_gamma_ccPairs * size_factor_view_ccPairs
    #         self.theta =  self.theta.to(self.device)
    #         dispersion_ccPairs = F.linear(one_hot_encoder(batch_ccPairs, self.n_conditions), self.theta)
    #         dispersion_ccPairs = torch.exp(dispersion_ccPairs)
    #         recon_loss_ccPairs = -nb(x=x_ccPairs, mu=dec_mean_ccPairs, theta=dispersion_ccPairs).sum(dim=-1).mean()
    #     
    #     mmd_loss = torch.tensor(0.0, device=z_concat.device)
    # 
    #     if self.use_mmd:
    #         #batch_concat = torch.concat((batch_ccPairs,batch_rest),dim=0)
    #         if self.mmd_on == "z":
    #             mmd_loss = mmd(z_ccPairs, batch_ccPairs,self.n_conditions, self.beta, self.mmd_boundary)
    #         else:
    #             mmd_loss = mmd(y1_ccPairs, batch_ccPairs,self.n_conditions, self.beta, self.mmd_boundary)
    # 
    #     return recon_loss_concat, kl_div_concat, mmd_loss
