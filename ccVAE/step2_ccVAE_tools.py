import numpy as np
import logging
from anndata import AnnData
import anndata as ad
from scipy.sparse import csr_matrix, hstack

import inspect
import os
import torch
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy
from anndata import AnnData, read
from typing import Optional, Union
from torch.distributions import Normal
from scipy.sparse import issparse
from itertools import zip_longest
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)



def _validate_var_names(adata, source_var_names):
    #Warning for gene percentage
    user_var_names = adata.var_names
    try:
        percentage = (len(user_var_names.intersection(source_var_names)) / len(user_var_names)) * 100
        percentage = round(percentage, 4)
        if percentage != 100:
            logger.warning(f"WARNING: Query shares {percentage}% of its genes with the reference."
                            "This may lead to inaccuracy in the results.")
    except Exception:
            logger.warning("WARNING: Something is wrong with the reference genes.")

    user_var_names = user_var_names.astype(str)
    new_adata = adata

    # Get genes in reference that are not in query
    ref_genes_not_in_query = []
    for name in source_var_names:
        if name not in user_var_names:
            ref_genes_not_in_query.append(name)

    if len(ref_genes_not_in_query) > 0:
        print("Query data is missing expression data of ",
              len(ref_genes_not_in_query),
              " genes which were contained in the reference dataset.")
        print("The missing information will be filled with zeroes.")
       
        filling_X = np.zeros((len(adata), len(ref_genes_not_in_query)))
        if isinstance(adata.X, csr_matrix): 
            filling_X = csr_matrix(filling_X) # support csr sparse matrix
            new_target_X = hstack((adata.X, filling_X))
        else:
            new_target_X = np.concatenate((adata.X, filling_X), axis=1)
        new_target_vars = adata.var_names.tolist() + ref_genes_not_in_query
        new_adata = AnnData(new_target_X, dtype="float32")
        new_adata.var_names = new_target_vars
        new_adata.obs = adata.obs.copy()

    if len(user_var_names) - (len(source_var_names) - len(ref_genes_not_in_query)) > 0:
        print(
            "Query data contains expression data of ",
            len(user_var_names) - (len(source_var_names) - len(ref_genes_not_in_query)),
            " genes that were not contained in the reference dataset. This information "
            "will be removed from the query data object for further processing.")

        # remove unseen gene information and order anndata
        new_adata = new_adata[:, source_var_names].copy()

    print(new_adata)

    return new_adata
  



class EarlyStopping(object):
    """Class for EarlyStopping. This class contains the implementation of early stopping for TRVAE/CVAE training.

       This early stopping class was inspired by:
       Title: scvi-tools
       Authors: Romain Lopez <romain_lopez@gmail.com>,
                Adam Gayoso <adamgayoso@berkeley.edu>,
                Galen Xing <gx2113@columbia.edu>
       Date: 24th December 2020
       Code version: 0.8.1
       Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/trainers/trainer.py

           Parameters
           ----------
           early_stopping_metric: : String
                The metric/loss which the early stopping criterion gets calculated on.
           threshold: Float
                The minimum value which counts as improvement.
           patience: Integer
                Number of epochs which are allowed to have no improvement until the training is stopped.
           reduce_lr: Boolean
                If 'True', the learning rate gets adjusted by 'lr_factor' after a given number of epochs with no
                improvement.
           lr_patience: Integer
                Number of epochs which are allowed to have no improvement until the learning rate is adjusted.
           lr_factor: Float
                Scaling factor for adjusting the learning rate.
        """
    def __init__(self,
                 early_stopping_metric: str = "val_unweighted_loss",
                 mode: str = "min",
                 threshold: float = 0,
                 patience: int = 20,
                 reduce_lr: bool = True,
                 lr_patience: int = 13,
                 lr_factor: float = 0.1):

        self.early_stopping_metric = early_stopping_metric
        self.mode = mode
        self.threshold = threshold
        self.patience = patience
        self.reduce_lr = reduce_lr
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor

        self.epoch = 0
        self.wait = 0
        self.wait_lr = 0
        self.current_performance = np.inf
        if self.mode == "min":
            self.best_performance = np.inf
            self.best_performance_state = np.inf
        else:
            self.best_performance = -np.inf
            self.best_performance_state = -np.inf

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, scalar):
        self.epoch += 1
        if self.epoch < self.patience:
            continue_training = True
            lr_update = False
        elif self.wait >= self.patience:
            continue_training = False
            lr_update = False
        else:
            if not self.reduce_lr:
                lr_update = False
            elif self.wait_lr >= self.lr_patience:
                lr_update = True
                self.wait_lr = 0
            else:
                lr_update = False
            # Shift
            self.current_performance = scalar
            if self.mode == "min":
                improvement = self.best_performance - self.current_performance
            else:
                improvement = self.current_performance - self.best_performance

            # updating best performance
            if improvement > 0:
                self.best_performance = self.current_performance

            if improvement < self.threshold:
                self.wait += 1
                self.wait_lr += 1
            else:
                self.wait = 0
                self.wait_lr = 0

            continue_training = True

        if not continue_training:
            print("\nStopping early: no improvement of more than " + str(self.threshold) +
                  " nats in " + str(self.patience) + " epochs")
            print("If the early stopping criterion is too strong, "
                  "please instantiate it with different parameters in the train method.")
        return continue_training, lr_update

    def update_state(self, scalar):
        if self.mode == "min":
            improved = (self.best_performance_state - scalar) > 0
        else:
            improved = (scalar - self.best_performance_state) > 0

        if improved:
            self.best_performance_state = scalar
        return improved
      



class BaseMixin:
    """ Adapted from
        Title: scvi-tools
        Authors: Romain Lopez <romain_lopez@gmail.com>,
                 Adam Gayoso <adamgayoso@berkeley.edu>,
                 Galen Xing <gx2113@columbia.edu>
        Date: 14.12.2020
        Code version: 0.8.0-beta.0
        Availability: https://github.com/YosefLab/scvi-tools
        Link to the used code:
        https://github.com/YosefLab/scvi-tools/blob/0.8.0-beta.0/scvi/core/models/base.py
    """
    def _get_user_attributes(self):
        # returns all the self attributes defined in a model class, eg, self.is_trained_
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [
            a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))
        ]
        attributes = [a for a in attributes if not a[0].startswith("_abc_")]
        return attributes

    def _get_public_attributes(self):
        public_attributes = self._get_user_attributes()
        public_attributes = {a[0]: a[1] for a in public_attributes if a[0][-1] == "_"}
        return public_attributes

    def plot_history(self, show=True, save=False, dir_path=None):
        if save:
            show = False
            if dir_path is None:
                save = False

        if self.trainer is None:
            print("Not possible if no trainer is provided")
            return
        fig = plt.figure()
        elbo_train = self.trainer.logs["epoch_loss"]
        elbo_test = self.trainer.logs["val_loss"]
        x = np.linspace(0, len(elbo_train), num=len(elbo_train))
        plt.plot(x, elbo_train, label="Train")
        plt.plot(x, elbo_test, label="Validate")
        plt.ylim(min(elbo_test) - 50, max(elbo_test) + 50)
        plt.legend()
        if save:
            plt.savefig(f'{dir_path}.png', bbox_inches='tight')
        if show:
            plt.show()
        plt.clf()
    
    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """Save the state of the model.
           Neither the trainer optimizer state nor the trainer history are saved.
           Parameters
           ----------
           dir_path
                Path to a directory.
           overwrite
                Overwrite existing data or not. If `False` and directory
                already exists at `dir_path`, error will be raised.
           save_anndata
                If True, also saves the anndata
           anndata_write_kwargs
                Kwargs for anndata write function
        """
        # get all the public attributes
        public_attributes = self._get_public_attributes()
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )

        # Concatenate the two AnnData objects
        adata_concat = ad.concat([self.adata_ccPairs, self.adata_rest], join='outer')
        
        if save_anndata:
            adata_concat.write(
                os.path.join(dir_path, "adata_concat.h5ad"), **anndata_write_kwargs
            )
            self.adata_ccPairs.write(
                os.path.join(dir_path, "adata_ccPairs.h5ad"), **anndata_write_kwargs
            )
            self.adata_rest.write(
                os.path.join(dir_path, "adata_rest.h5ad"), **anndata_write_kwargs
            )
            
        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")
        varnames_save_path = os.path.join(dir_path, "var_names.csv")

        var_names = adata_concat.var_names.astype(str)
        var_names = var_names.to_numpy()
        np.savetxt(varnames_save_path, var_names, fmt="%s")

        torch.save(self.model.state_dict(), model_save_path)
        with open(attr_save_path, "wb") as f:
            pickle.dump(public_attributes, f)

    def _load_expand_params_from_dict(self, state_dict):
        load_state_dict = state_dict.copy()

        device = next(self.model.parameters()).device

        new_state_dict = self.model.state_dict()
        for key, load_ten in load_state_dict.items():
            new_ten = new_state_dict[key]
            if new_ten.size() == load_ten.size():
                continue
            # new categoricals changed size
            else:
                load_ten = load_ten.to(device)
                # only one dim diff
                new_shape = new_ten.shape
                n_dims = len(new_shape)
                sel = [slice(None)] * n_dims
                for i in range(n_dims):
                    dim_diff = new_shape[i] - load_ten.shape[i]
                    axs = i
                    sel[i] = slice(-dim_diff, None)
                    if dim_diff > 0:
                        break
                fixed_ten = torch.cat([load_ten, new_ten[tuple(sel)]], dim=axs)
                load_state_dict[key] = fixed_ten

        for key, ten in new_state_dict.items():
            if key not in load_state_dict:
                load_state_dict[key] = ten

        self.model.load_state_dict(load_state_dict)

    @classmethod
    def _load_params(cls, dir_path: str, map_location: Optional[str] = None):
        setup_dict_path = os.path.join(dir_path, "attr.pkl")
        model_path = os.path.join(dir_path, "model_params.pt")
        varnames_path = os.path.join(dir_path, "var_names.csv")

        with open(setup_dict_path, "rb") as handle:
            attr_dict = pickle.load(handle)

        model_state_dict = torch.load(model_path, map_location=map_location)

        var_names = np.genfromtxt(varnames_path, delimiter=",", dtype=str)

        return attr_dict, model_state_dict, var_names

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata_ccPairs: Optional[AnnData] = None,
        adata_rest: Optional[AnnData] = None,
        map_location = None
    ):
        """Instantiate a model from the saved output.
           Parameters
           ----------
           dir_path
                Path to saved outputs.
           adata
                AnnData object.
                If None, will check for and load anndata saved with the model.
           map_location
                 a function, torch.device, string or a dict specifying
                 how to remap storage locations
           Returns
           -------
                Model with loaded state dictionaries.
        """
        adata_path_ccPairs = os.path.join(dir_path, "adata_ccPairs.h5ad")
        adata_path_rest = os.path.join(dir_path, "adata_rest.h5ad")

        load_adata_ccPairs = adata_ccPairs is None
        load_adata_rest = adata_rest is None

        if os.path.exists(adata_path_ccPairs) and load_adata_ccPairs and os.path.exists(adata_path_rest) and load_adata_rest:
            adata_ccPairs = read(adata_path_ccPairs)
            adata_rest = read(adata_path_rest)
        elif not os.path.exists(adata_path_ccPairs) and load_adata_ccPairs:
            raise ValueError("Save path contains no saved anndata and no adata was passed.")

        attr_dict, model_state_dict, var_names = cls._load_params(dir_path, map_location)

        # Overwrite adata with new genes
        adata_ccPairs = _validate_var_names(adata_ccPairs, var_names)
        adata_rest = _validate_var_names(adata_rest, var_names)

        cls._validate_adata(adata_ccPairs, attr_dict, use_ccPairs='ccPairs')
        cls._validate_adata(adata_rest, attr_dict, use_ccPairs='rest')
        init_params = cls._get_init_params_from_dict(attr_dict)

        model = cls(adata_ccPairs, adata_rest, **init_params)
        model.model.to(next(iter(model_state_dict.values())).device)
        model.model.load_state_dict(model_state_dict)
        model.model.eval()

        model.is_trained_ = attr_dict['is_trained_']

        return model


class SurgeryMixin:
    @classmethod
    def load_query_data(
        cls,
        adata_ccPairs: AnnData,
        adata_query: AnnData,
        reference_model: Union[str, 'Model'],
        freeze: bool = True,
        freeze_expression: bool = True,
        remove_dropout: bool = True,
        map_location = None,
        **kwargs
    ):
        """Transfer Learning function for new data. Uses old trained model and expands it for new conditions.

           Parameters
           ----------
           adata_ccPairs
                ccPairs anndata object.
           adata_query
                Query anndata object.
           reference_model
                A model to expand or a path to a model folder.
           freeze: Boolean
                If 'True' freezes every part of the network except the first layers of encoder/decoder.
           freeze_expression: Boolean
                If 'True' freeze every weight in first layers except the condition weights.
           remove_dropout: Boolean
                If 'True' remove Dropout for Transfer Learning.
           map_location
                map_location to remap storage locations (as in '.load') of 'reference_model'.
                Only taken into account if 'reference_model' is a path to a model on disk.
           kwargs
                kwargs for the initialization of the query model.

           Returns
           -------
           new_model
                New model to train on query data.
        """
        adata_path_ccPairs = os.path.join(reference_model, "adata_ccPairs.h5ad")

        load_adata_ccPairs = adata_ccPairs is None

        if os.path.exists(adata_path_ccPairs) and load_adata_ccPairs:
            adata_ccPairs = read(adata_path_ccPairs)
        elif not os.path.exists(adata_path_ccPairs) and load_adata_ccPairs:
            raise ValueError("Save path contains no saved anndata and no adata was passed.")
        
        if isinstance(reference_model, str):
            attr_dict, model_state_dict, var_names = cls._load_params(reference_model, map_location)
            #adata_ccPairs = _validate_var_names(adata_ccPairs, var_names)
            adata_query = _validate_var_names(adata_query, var_names)
        else:
            attr_dict = reference_model._get_public_attributes()
            model_state_dict = reference_model.model.state_dict()
            adata_query = _validate_var_names(adata_query, reference_model.adata_ccPairs.var_names)

        init_params = deepcopy(cls._get_init_params_from_dict(attr_dict))

        conditions = init_params['conditions']
        condition_key = init_params['condition_key']

        new_conditions = []
        adata_conditions = adata_query.obs[condition_key].unique().tolist()
        # Check if new conditions are already known
        for item in adata_conditions:
            if item not in conditions:
                new_conditions.append(item)

        # Add new conditions to overall conditions
        for condition in new_conditions:
            conditions.append(condition)

        if remove_dropout:
            init_params['dr_rate'] = 0.0

        init_params.update(kwargs)

        new_model = cls(adata_ccPairs,adata_query, **init_params)
        new_model.model.to(next(iter(model_state_dict.values())).device)
        new_model._load_expand_params_from_dict(model_state_dict)

        if freeze:
            new_model.model.freeze = True
            for name, p in new_model.model.named_parameters():
                p.requires_grad = False
                if 'theta' in name:
                    p.requires_grad = True
                if freeze_expression:
                    if 'cond_L.weight' in name:
                        p.requires_grad = True
                else:
                    if "L0" in name or "N0" in name:
                        p.requires_grad = True

        return new_model



class CVAELatentsMixin:
    def get_latent(
        self,
        x_ccPairs: Optional[np.ndarray] = None,
        x_rest: Optional[np.ndarray] = None,
        batch_ccPairs: Optional[np.ndarray] = None,
        batch_rest: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get the latent representations for the given input data.
    
        Parameters:
            x_ccPairs (np.ndarray, optional): Input data for ccPairs. Shape: [n_samples_ccPairs, input_dim_ccPairs].
                Defaults to None.
            x_rest (np.ndarray, optional): Input data for rest. Shape: [n_samples_rest, input_dim_rest].
                Defaults to None.
            batch_ccPairs (np.ndarray, optional): Batch labels for ccPairs data. Shape: [n_samples_ccPairs].
                Defaults to None.
            batch_rest (np.ndarray, optional): Batch labels for rest data. Shape: [n_samples_rest].
                Defaults to None.
    
        Returns:
            np.ndarray: Latent representations. Shape: [n_samples_ccPairs + n_samples_rest, latent_dim].
        """
        device = next(self.model.parameters()).device
    
        if x_ccPairs is None and x_rest is None and batch_ccPairs is None and batch_rest is None:
            x_ccPairs = self.adata_ccPairs.X
            x_rest = self.adata_rest.X
            if self.conditions_ is not None:
                batch_ccPairs = self.adata_ccPairs.obs[self.condition_key_]
                batch_rest = self.adata_rest.obs[self.condition_key_]
    
        batch_ccPairs = self._process_conditions(batch_ccPairs, self.model.condition_encoder, device)
        batch_rest = self._process_conditions(batch_rest, self.model.condition_encoder, device)
    
        if x_ccPairs is not None:
            x_ccPairs = torch.tensor(x_ccPairs, device=device)
            latents_ccPairs = self.model.vae1.get_latent(x_ccPairs, batch_ccPairs)
    
        if x_rest is not None:
            x_rest = torch.tensor(x_rest, device=device)
            latents_rest = self.model.vae2.get_latent(x_rest, batch_rest)
    
        result = torch.cat((latents_ccPairs, latents_rest), dim=0).cpu().detach().numpy()
        result = result[0] if len(result) == 1 else result
    
        return result


    def get_y(
        self,
        x_ccPairs: Optional[np.ndarray] = None,
        x_rest: Optional[np.ndarray] = None,
        batch_ccPairs: Optional[np.ndarray] = None,
        batch_rest: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get the output of the first layer of the decoder (y dimension) for the given input data.

        Parameters:
            x_ccPairs (np.ndarray, optional): Input data for ccPairs. Shape: [n_samples_ccPairs, input_dim_ccPairs].
                Defaults to None.
            x_rest (np.ndarray, optional): Input data for rest. Shape: [n_samples_rest, input_dim_rest].
                Defaults to None.
            batch_ccPairs (np.ndarray, optional): Batch labels for ccPairs data. Shape: [n_samples_ccPairs].
                Defaults to None.
            batch_rest (np.ndarray, optional): Batch labels for rest data. Shape: [n_samples_rest].
                Defaults to None.

        Returns:
            np.ndarray: Output of the first decoder layer. Shape: [n_samples_ccPairs + n_samples_rest, y_dim].
        """
        device = next(self.model.parameters()).device

        if x_ccPairs is None and x_rest is None and batch_ccPairs is None and batch_rest is None:
            x_ccPairs = self.adata_ccPairs.X
            x_rest = self.adata_rest.X
            if self.conditions_ is not None:
                batch_ccPairs = self.adata_ccPairs.obs[self.condition_key_]
                batch_rest = self.adata_rest.obs[self.condition_key_]

        batch_ccPairs = self._process_conditions(batch_ccPairs, self.model.condition_encoder, device)
        batch_rest = self._process_conditions(batch_rest, self.model.condition_encoder, device)

        if x_ccPairs is not None:
            x_ccPairs = torch.tensor(x_ccPairs, device=device)

        if x_rest is not None:
            x_rest = torch.tensor(x_rest, device=device)

        y1_concat = self.model.get_y(x_ccPairs, x_rest, batch_ccPairs, batch_rest)

        result = y1_concat.cpu().detach().numpy()
        result = result[0] if len(result) == 1 else result

        return result

    
    def label_transfer(self,
                       ref_ccPairs: Optional[np.ndarray] = None,
                       query_rest: Optional[np.ndarray] = None,
                       rep='latent', 
                       label='celltype'):
        """
        Label transfer

        Parameters
        -----------
        ref_ccPairs
            reference containing the projected representations and labels
        query_rest
            query data to transfer label
        rep
            representations to train the classifier. Default is `latent`
        label
            label name. Defautl is `celltype` stored in ref.obs

        Returns
        --------
        transfered label
        """
        device = next(self.model.parameters()).device
        if ref_ccPairs is not None:
            ref_ccPairs2 = torch.tensor(ref_ccPairs.X, device=device)
        else:
            ValueError("Not found reference adata")

        if query_rest is not None:
            query_rest2 = torch.tensor(query_rest.X, device=device)
        else:
            ValueError("Not found query adata")    
        
        # get condition info
        batch_ccPairs = self.adata_ccPairs.obs[self.condition_key_]
        batch_ccPairs = self._process_conditions(batch_ccPairs, self.model.condition_encoder, device)
        batch_rest = query_rest.obs[self.condition_key_] 
        batch_rest = self._process_conditions(batch_rest, self.model.condition_encoder, device)
        
        ## get latents
        latents_ccPairs = self.model.vae1.get_latent(ref_ccPairs2, batch_ccPairs)
        latents_rest = self.model.vae2.get_latent(query_rest2, batch_rest)
        
        ## save latent rep
        ref_ccPairs.obsm[rep] = latents_ccPairs.cpu().detach().numpy() 
        query_rest.obsm[rep] = latents_rest.cpu().detach().numpy()
        
        X_train = ref_ccPairs.obsm[rep]
        y_train = ref_ccPairs.obs[label]
        X_test = query_rest.obsm[rep]

        knn = KNeighborsClassifier().fit(X_train, y_train)
        y_test = knn.predict(X_test)
        return y_test
    
    
    @staticmethod
    def _process_conditions(conditions, condition_encoder, device):
        if conditions is not None:
            conditions = np.asarray(conditions)
            if not set(conditions).issubset(condition_encoder):
                raise ValueError("Incorrect conditions")
            labels = np.zeros(conditions.shape[0])
            for condition, label in condition_encoder.items():
                labels[conditions == condition] = label
            conditions = torch.tensor(labels, device=device)
        return conditions
