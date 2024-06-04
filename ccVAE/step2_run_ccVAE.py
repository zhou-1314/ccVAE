import inspect
import os
import torch
import pickle
import numpy as np

import anndata as ad
from anndata import AnnData, read
from copy import deepcopy
from typing import Optional, Union

from .step2_ccVAE_MainModel import ccVAE
from .step2_ccVAE_Trainer import ccVAETrainer
from .step2_ccVAE_tools import _validate_var_names
from .step2_ccVAE_tools import BaseMixin, SurgeryMixin, CVAELatentsMixin


class CCVAE(BaseMixin, SurgeryMixin, CVAELatentsMixin):
    """Model for scArches class. This class contains the implementation of Conditional Variational Auto-encoder.

       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix. Has to be count data for 'nb' and 'zinb' loss and normalized log transformed data
            for 'mse' loss.
       condition_key: String
            column name of conditions in `adata.obs` data frame.
       conditions: List
            List of Condition names that the used data will contain to get the right encoding when used after reloading.
       cell_type_key: String
            column name of cell type in `adata.obs` data frame.
       hidden_layer_sizes: List
            A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
       latent_dim: Integer
            Bottleneck layer (z)  size.
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
            Scaling Factor for MMD loss
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
    """
    def __init__(
        self,
        adata_ccPairs: AnnData,
        adata_rest: AnnData,
        condition_key: str = None,
        conditions: Optional[list] = None,
        cell_type_key: str = None,
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
        self.adata_ccPairs = adata_ccPairs
        self.adata_rest = adata_rest
        self.condition_encoder = {}
        self.condition_encoder = {k: v for k, v in zip(conditions, range(len(conditions)))}
        self.condition_key_ = condition_key
        self.freeze = None

        if conditions is None:
            if condition_key is not None:
                self.conditions_ = adata.obs[condition_key].unique().tolist()
            else:
                self.conditions_ = []
        else:
            self.conditions_ = conditions
            
        if cell_type_key is not None:
            # Concatenate the two AnnData objects
            adata = ad.concat([self.adata_ccPairs, self.adata_rest], join='outer')

            cell_type = adata.obs[cell_type_key].unique().tolist()
            number_of_class = len(adata.obs[cell_type_key].unique().tolist())
            self.number_of_class_ = number_of_class
            cell_type_encoder = {k: v for k, v in zip(cell_type, range(number_of_class))}
            cell_type_encoder_list = {i: cell_type_encoder[cell_type] if cell_type in cell_type_encoder else -1 for i, cell_type in enumerate(adata.obs[cell_type_key])}
            self.cell_type_encoder_ = cell_type_encoder_list
            del adata
        else:
            self.cell_type_encoder_ = None
            self.number_of_class_ = None

        self.hidden_layer_sizes_ = hidden_layer_sizes
        self.latent_dim_ = latent_dim
        self.use_ccPairs_ = use_ccPairs
        self.dr_rate_ = dr_rate
        self.use_mmd_ = use_mmd
        self.mmd_on_ = mmd_on
        self.mmd_boundary_ = mmd_boundary
        self.recon_loss_ = recon_loss
        self.beta_ = beta
        self.use_bn_ = use_bn
        self.use_ln_ = use_ln

        self.input_dim1_ = adata_ccPairs.n_vars
        self.input_dim2_ = adata_rest.n_vars

        self.model = ccVAE(
            self.input_dim1_,
            self.input_dim2_,
            self.condition_key_,
            self.conditions_,
            self.cell_type_encoder_,
            self.number_of_class_,
            self.hidden_layer_sizes_,
            self.latent_dim_,
            self.use_ccPairs_,
            self.dr_rate_,
            self.use_mmd_,
            self.mmd_on_,
            self.mmd_boundary_,
            self.recon_loss_,
            self.beta_,
            self.use_bn_,
            self.use_ln_,
        )
        

        self.is_trained_ = False

        self.trainer = None

    def train(
        self,
        n_epochs: int = 400,
        lr: float = 1e-3,
        eps: float = 0.01,
        **kwargs
    ):
        """Train the model.

           Parameters
           ----------
           n_epochs
                Number of epochs for training the model.
           lr
                Learning rate for training the model.
           eps
                torch.optim.Adam eps parameter
           kwargs
                kwargs for the ccVAE trainer.
        """
        self.trainer = ccVAETrainer(
            self.model,
            self.adata_ccPairs,
            self.adata_rest,
            condition_key=self.condition_key_,
            **kwargs)
        self.trainer.train(n_epochs, lr, eps)
        self.is_trained_ = True

    @classmethod
    def _get_init_params_from_dict(cls, dct):
        init_params = {
            'condition_key': dct['condition_key_'],
            'conditions': dct['conditions_'],
            'cell_type_key': dct['cell_type_key'],
            'hidden_layer_sizes': dct['hidden_layer_sizes_'],
            'latent_dim': dct['latent_dim_'],
            'use_ccPairs': dct['use_ccPairs_'],
            'dr_rate': dct['dr_rate_'],
            'use_mmd': dct['use_mmd_'],
            'mmd_on': dct['mmd_on_'],
            'mmd_boundary': dct['mmd_boundary_'],
            'recon_loss': dct['recon_loss_'],
            'beta': dct['beta_'],
            'use_bn': dct['use_bn_'],
            'use_ln': dct['use_ln_'],
        }

        return init_params

    @classmethod
    def _validate_adata(cls, adata, dct, use_ccPairs=None):
        if use_ccPairs == 'ccPairs':
            if adata.n_vars != dct['input_dim1_']:
                raise ValueError("Incorrect var dimension")
        else:
            if adata.n_vars != dct['input_dim2_']:
                raise ValueError("Incorrect var dimension")

        adata_conditions = adata.obs[dct['condition_key_']].unique().tolist()
        if not set(adata_conditions).issubset(dct['conditions_']):
            raise ValueError("Incorrect conditions")
          
        adata_conditions = adata.obs[dct['cell_type_key']].unique().tolist()
        if not set(adata_conditions).issubset(dct['cell_type_key']):
            raise ValueError("Incorrect cell_type")
