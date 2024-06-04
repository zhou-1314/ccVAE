import sys
import time
import re
import numpy as np
import torch
import torch.nn as nn
import collections.abc as container_abcs
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from .step2_ccVAE_tools import EarlyStopping
from .step2_ccVAE_dataloaders import ccVAEDataset



class ccVAETrainer:
    """ccVAE base Trainer class. This class contains the implementation of the base CVAE/TRVAE Trainer.

       Parameters
       ----------
       model: ccVAE
            Number of input features (i.e. gene in case of scRNA-seq).
       adata: : `~anndata.AnnData`
            Annotated data matrix. Has to be count data for 'nb' and 'zinb' loss and normalized log transformed data
            for 'mse' loss.
       condition_key: String
            column name of conditions in `adata.obs` data frame.
       cell_type_keys: List
            List of column names of different celltype levels in `adata.obs` data frame.
       batch_size: Integer
            Defines the batch size that is used during each Iteration
       alpha_epoch_anneal: Integer or None
            If not 'None', the KL Loss scaling factor (alpha_kl) will be annealed from 0 to 1 every epoch until the input
            integer is reached.
       alpha_kl: Float
            Multiplies the KL divergence part of the loss.
       alpha_iter_anneal: Integer or None
            If not 'None', the KL Loss scaling factor will be annealed from 0 to 1 every iteration until the input
            integer is reached.
       use_early_stopping: Boolean
            If 'True' the EarlyStopping class is being used for training to prevent overfitting.
       reload_best: Boolean
            If 'True' the best state of the model during training concerning the early stopping criterion is reloaded
            at the end of training.
       early_stopping_kwargs: Dict
            Passes custom Earlystopping parameters.
       train_frac: Float
            Defines the fraction of data that is used for training and data that is used for validation.
       n_samples: Integer or None
            Defines how many samples are being used during each epoch. This should only be used if hardware resources
            are limited.
       use_stratified_sampling: Boolean
            If 'True', the sampler tries to load equally distributed batches concerning the conditions in every
            iteration.
       monitor: Boolean
            If `True', the progress of the training will be printed after each epoch.
       monitor_only_val: Boolean
            If `True', only the progress of the validation dataset is displayed.
       clip_value: Float
            If the value is greater than 0, all gradients with an higher value will be clipped during training.
       weight decay: Float
            Defines the scaling factor for weight decay in the Adam optimizer.
       n_workers: Integer
            Passes the 'n_workers' parameter for the torch.utils.data.DataLoader class.
       seed: Integer
            Define a specific random seed to get reproducable results.
    """
    def __init__(self,
                 model,
                 adata_ccPairs,
                 adata_rest,
                 condition_key: str = None,
                 cell_type_keys: str = None,
                 batch_size: int = 128,
                 alpha_epoch_anneal: int = None,
                 alpha_kl: float = 1.,
                 use_early_stopping: bool = True,
                 reload_best: bool = True,
                 early_stopping_kwargs: dict = None,
                 **kwargs):

        self.adata_ccPairs = adata_ccPairs
        self.adata_rest = adata_rest
        self.model = model
        self.condition_key = condition_key
        self.cell_type_keys = cell_type_keys

        self.batch_size = batch_size
        self.alpha_epoch_anneal = alpha_epoch_anneal
        self.alpha_iter_anneal = kwargs.pop("alpha_iter_anneal", None)
        self.use_early_stopping = use_early_stopping
        self.reload_best = reload_best

        self.alpha_kl = alpha_kl

        early_stopping_kwargs = (early_stopping_kwargs if early_stopping_kwargs else dict())

        self.n_samples = kwargs.pop("n_samples", None)
        self.train_frac = kwargs.pop("train_frac", 0.9)
        self.use_stratified_sampling = kwargs.pop("use_stratified_sampling", True)

        self.weight_decay = kwargs.pop("weight_decay", 0.04)
        self.clip_value = kwargs.pop("clip_value", 0.0)

        self.n_workers = kwargs.pop("n_workers", 0)
        self.seed = kwargs.pop("seed", 2023)
        self.monitor = kwargs.pop("monitor", True)
        self.monitor_only_val = kwargs.pop("monitor_only_val", True)

        self.early_stopping = EarlyStopping(**early_stopping_kwargs)

        torch.manual_seed(self.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            self.model.cuda()
        
        self.epoch = -1
        self.n_epochs = None
        self.iter = 0
        self.best_epoch = None
        self.best_state_dict = None
        self.current_loss = None
        self.previous_loss_was_nan = False
        self.nan_counter = 0
        self.optimizer = None
        self.training_time = 0

        self.train_data_ccPairs = None
        self.valid_data_ccPairs = None
        self.train_data_rest = None
        self.valid_data_rest = None
        self.sampler = None
        self.dataloader_train_ccPairs = None
        self.dataloader_valid_ccPairs = None
        self.dataloader_train_rest = None
        self.dataloader_valid_rest = None

        self.iters_per_epoch_ccPairs = None
        self.iters_per_epoch_rest = None
        self.val_iters_per_epoch = None

        self.logs = defaultdict(list)

        # Create Train/Valid AnnotatetDataset objects for ccPairs cells
        self.train_data_ccPairs, self.valid_data_ccPairs, self.train_data_rest, self.valid_data_rest = make_dataset(
            self.adata_ccPairs,
            self.adata_rest,
            train_frac=self.train_frac,
            condition_key=self.condition_key,
            cell_type_keys=self.cell_type_keys,
            condition_encoder=self.model.condition_encoder,
            cell_type_encoder=self.model.cell_type_encoder,
        )
        

    def initialize_loaders(self):
        """
        Initializes Train-/Test Data and Dataloaders with custom_collate and WeightedRandomSampler for Trainloader.
        Returns:

        """
        if self.n_samples is None or self.n_samples > len(self.train_data_ccPairs):
            self.n_samples_ccPairs = len(self.train_data_ccPairs)
        self.iters_per_epoch_ccPairs = int(np.ceil(self.n_samples_ccPairs / self.batch_size))

        if self.use_stratified_sampling:
            # Create Sampler and Dataloaders
            stratifier_weights = torch.tensor(self.train_data_ccPairs.stratifier_weights, device=self.device)

            self.sampler = WeightedRandomSampler(stratifier_weights,
                                                 num_samples=self.n_samples_ccPairs,
                                                 replacement=True)
            self.dataloader_train_ccPairs = torch.utils.data.DataLoader(dataset=self.train_data_ccPairs,
                                                                batch_size=self.batch_size,
                                                                sampler=self.sampler,
                                                                collate_fn=custom_collate,
                                                                num_workers=self.n_workers)
        else:
            self.dataloader_train_ccPairs = torch.utils.data.DataLoader(dataset=self.train_data_ccPairs,
                                                                batch_size=self.batch_size,
                                                                shuffle=True,
                                                                collate_fn=custom_collate,
                                                                num_workers=self.n_workers)
        if self.valid_data_ccPairs is not None:
            val_batch_size = self.batch_size
            if self.batch_size > len(self.valid_data_ccPairs):
                val_batch_size = len(self.valid_data_ccPairs)
            self.val_iters_per_epoch = int(np.ceil(len(self.valid_data_ccPairs) / self.batch_size))
            self.dataloader_valid_ccPairs = torch.utils.data.DataLoader(dataset=self.valid_data_ccPairs,
                                                                batch_size=val_batch_size,
                                                                shuffle=True,
                                                                collate_fn=custom_collate,
                                                                num_workers=self.n_workers)
            
        ### rest cells
        if self.n_samples is None or self.n_samples > len(self.train_data_rest):
            self.n_samples_rest = len(self.train_data_rest)
        self.iters_per_epoch_rest = int(np.ceil(self.n_samples_rest / self.batch_size))

        if self.use_stratified_sampling:
            # Create Sampler and Dataloaders
            stratifier_weights = torch.tensor(self.train_data_rest.stratifier_weights, device=self.device)

            self.sampler = WeightedRandomSampler(stratifier_weights,
                                                 num_samples=self.n_samples_rest,
                                                 replacement=True)
            self.dataloader_train_rest = torch.utils.data.DataLoader(dataset=self.train_data_rest,
                                                                batch_size=self.batch_size,
                                                                sampler=self.sampler,
                                                                collate_fn=custom_collate,
                                                                num_workers=self.n_workers)
        else:
            self.dataloader_train_rest = torch.utils.data.DataLoader(dataset=self.train_data_rest,
                                                                batch_size=self.batch_size,
                                                                shuffle=True,
                                                                collate_fn=custom_collate,
                                                                num_workers=self.n_workers)
        if self.valid_data_rest is not None:
            val_batch_size = self.batch_size
            if self.batch_size > len(self.valid_data_rest):
                val_batch_size = len(self.valid_data_rest)
            self.val_iters_per_epoch = int(np.ceil(len(self.valid_data_rest) / self.batch_size))
            self.dataloader_valid_rest = torch.utils.data.DataLoader(dataset=self.valid_data_rest,
                                                                batch_size=val_batch_size,
                                                                shuffle=True,
                                                                collate_fn=custom_collate,
                                                                num_workers=self.n_workers)

    def calc_alpha_coeff_ccPairs(self):
        """Calculates current alpha coefficient for alpha annealing.

           Parameters
           ----------

           Returns
           -------
           Current annealed alpha value
        """
        if self.alpha_epoch_anneal is not None:
            alpha_coeff = min(self.alpha_kl * self.epoch / self.alpha_epoch_anneal, self.alpha_kl)
        elif self.alpha_iter_anneal is not None:
            alpha_coeff = min((self.alpha_kl * (self.epoch * self.iters_per_epoch_ccPairs + self.iter) / self.alpha_iter_anneal), self.alpha_kl)
        else:
            alpha_coeff = self.alpha_kl
        return alpha_coeff
    
    
    def train(self,
              n_epochs=400,
              lr=1e-3,
              eps=0.01):

        self.initialize_loaders()
        begin = time.time()
        self.model.train() # 将模型设置为训练状态
        self.n_epochs = n_epochs

        params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps, weight_decay=self.weight_decay)

        self.before_loop()

        ## train of ccPair and rest cells
        for self.epoch in range(n_epochs):
            self.on_epoch_begin(lr, eps)
            self.iter_logs = defaultdict(list)
    
            for self.iter, (batch_ccPairs, batch_rest) in enumerate(zip(self.dataloader_train_ccPairs, self.dataloader_train_rest)):
                merged_batch = { **batch_ccPairs, **batch_rest }
                for key, batch in merged_batch.items():
                    merged_batch[key] = batch.to(self.device)

                # Loss Calculation
                self.on_iteration(merged_batch)

            # Validation of Model, Monitoring, Early Stopping
            self.on_epoch_end()

            if self.use_early_stopping:
                if not self.check_early_stop():
                    break

        if self.best_state_dict is not None and self.reload_best:
            print("Saving best state of network...")
            print("Best State was in Epoch", self.best_epoch)
            self.model.load_state_dict(self.best_state_dict)

        self.model.eval()
        self.after_loop()

        self.training_time += (time.time() - begin)
        
    
    def before_loop(self):
        pass

    def on_epoch_begin(self, lr, eps):
        pass

    def after_loop(self):
        pass
      
    def calculate_loss(self, total_batch=None):
        if self.model.cell_type_encoder is not None:
            # return recon_loss_ccPairs, recon_loss_rest, kl_div_ccPairs, kl_div_rest, mmd_loss
            recon_loss_mean, kl_loss, mmd_loss, modality_recon_losses, center_loss_mean, modality_center_loss = self.model(**total_batch)
            kl_loss_coeff = self.calc_alpha_coeff_ccPairs()*kl_loss
            ccPairs_loss_mean = modality_recon_losses[0]
            rest_loss_mean = modality_recon_losses[1]
            ccPairs_center_loss_mean = modality_center_loss[0]
            rest_center_loss_mean = modality_center_loss[1]
            loss = recon_loss_mean + kl_loss_coeff + mmd_loss  + center_loss_mean + ccPairs_loss_mean + rest_loss_mean + ccPairs_center_loss_mean + rest_center_loss_mean

            self.iter_logs["loss"].append(loss.item())
            self.iter_logs["unweighted_loss"].append(recon_loss_mean.item() + kl_loss.item()  + mmd_loss.item() + center_loss_mean.item() + 
                                                     ccPairs_loss_mean.item() + rest_loss_mean.item() + ccPairs_center_loss_mean.item() + rest_center_loss_mean.item()) 
            self.iter_logs["recon_loss"].append(recon_loss_mean.item())
            self.iter_logs["kl_loss"].append(kl_loss.item())
            self.iter_logs["ccPairs_loss"].append(ccPairs_loss_mean.item())
            self.iter_logs["rest_loss"].append(rest_loss_mean.item())
            self.iter_logs["center_loss"].append(center_loss_mean.item())
            self.iter_logs["ccPairs_center_loss"].append(ccPairs_center_loss_mean.item())
            self.iter_logs["rest_center_loss"].append(rest_center_loss_mean.item())
        else:
            recon_loss_mean, kl_loss, mmd_loss, modality_recon_losses = self.model(**total_batch)
            kl_loss_coeff = self.calc_alpha_coeff_ccPairs()*kl_loss
            ccPairs_loss_mean = modality_recon_losses[0]
            rest_loss_mean = modality_recon_losses[1]
            loss = recon_loss_mean + kl_loss_coeff + mmd_loss  + ccPairs_loss_mean + rest_loss_mean

            self.iter_logs["loss"].append(loss.item())
            self.iter_logs["unweighted_loss"].append(recon_loss_mean.item() + kl_loss.item() + mmd_loss.item() + ccPairs_loss_mean.item() + rest_loss_mean.item()) 
            self.iter_logs["recon_loss"].append(recon_loss_mean.item())
            self.iter_logs["kl_loss"].append(kl_loss.item())
            self.iter_logs["ccPairs_loss"].append(ccPairs_loss_mean.item())
            self.iter_logs["rest_loss"].append(rest_loss_mean.item())

        if self.model.use_mmd:
            self.iter_logs["mmd_loss"].append(mmd_loss.item())

        return loss

    
    def on_iteration(self, merged_batch):
        # Dont update any weight on first layers except condition weights
        if self.model.freeze:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    if not module.weight.requires_grad:
                        module.affine = False
                        module.track_running_stats = False

        # Calculate Loss depending on Trainer/Model
        self.current_loss = loss = self.calculate_loss(merged_batch)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        if self.clip_value > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)

        self.optimizer.step()

        
    def on_epoch_end(self):
        # Get Train Epoch Logs
        for key in self.iter_logs:
            self.logs["epoch_" + key].append(np.array(self.iter_logs[key]).mean())

        # Validate Model
        if self.valid_data_ccPairs is not None and self.valid_data_rest is not None:
            self.validate()

        # Monitor Logs
        if self.monitor:
            print_progress(self.epoch, self.logs, self.n_epochs, self.monitor_only_val)

            
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.iter_logs = defaultdict(list)
        
        # Calculate Validation Losses
        for self.iter, (batch_ccPairs, batch_rest) in enumerate(zip(self.dataloader_valid_ccPairs, self.dataloader_valid_rest)):
                merged_batch = { **batch_ccPairs, **batch_rest }
                for key, batch in merged_batch.items():
                    merged_batch[key] = batch.to(self.device)

                val_loss = self.calculate_loss(merged_batch)
    
        # Get Validation Logs
        for key in self.iter_logs:
            self.logs["val_" + key].append(np.array(self.iter_logs[key]).mean())
            
        self.model.train()


    def check_early_stop(self):
        # Calculate Early Stopping and best state
        early_stopping_metric = self.early_stopping.early_stopping_metric
        if self.early_stopping.update_state(self.logs[early_stopping_metric][-1]):
            self.best_state_dict = self.model.state_dict()
            self.best_epoch = self.epoch

        continue_training, update_lr = self.early_stopping.step(self.logs[early_stopping_metric][-1])
        if update_lr:
            print(f'\nADJUSTED LR')
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.early_stopping.lr_factor

        return continue_training



def print_progress(epoch, logs, n_epochs=10000, only_val_losses=True):
    """Creates Message for '_print_progress_bar'.

       Parameters
       ----------
       epoch: Integer
            Current epoch iteration.
       logs: Dict
            Dictionary of all current losses.
       n_epochs: Integer
            Maximum value of epochs.
       only_val_losses: Boolean
            If 'True' only the validation dataset losses are displayed, if 'False' additionally the training dataset
            losses are displayed.

       Returns
       -------
    """
    message = ""
    for key in logs:
        if only_val_losses:
            if "val_" in key and "unweighted" not in key:
                message += f" - {key:s}: {logs[key][-1]:7.10f}"
        else:
            if "unweighted" not in key:
                message += f" - {key:s}: {logs[key][-1]:7.10f}"

    _print_progress_bar(epoch + 1, n_epochs, prefix='', suffix=message, decimals=1, length=20)


def _print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """Prints out message with a progress bar.

       Parameters
       ----------
       iteration: Integer
            Current epoch.
       total: Integer
            Maximum value of epochs.
       prefix: String
            String before the progress bar.
       suffix: String
            String after the progress bar.
       decimals: Integer
            Digits after comma for all the losses.
       length: Integer
            Length of the progress bar.
       fill: String
            Symbol for filling the bar.

       Returns
       -------
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_len = int(length * iteration // total)
    bar = fill * filled_len + '-' * (length - filled_len)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def train_test_split(adata, train_frac=0.85, condition_key=None, cell_type_key=None, labeled_array=None):
    """Splits 'Anndata' object into training and validation data.

       Parameters
       ----------
       adata: `~anndata.AnnData`
            `AnnData` object for training the model.
       train_frac: float
            Train-test split fraction. the model will be trained with train_frac for training
            and 1-train_frac for validation.

       Returns
       -------
       Indices for training and validating the model.
    """
    indices = np.arange(adata.shape[0])

    if train_frac == 1:
        return indices, None

    if cell_type_key is not None:
        labeled_array = np.zeros((len(adata), 1)) if labeled_array is None else labeled_array
        labeled_array = np.ravel(labeled_array)

        labeled_idx = indices[labeled_array == 1]
        unlabeled_idx = indices[labeled_array == 0]

        train_labeled_idx = []
        val_labeled_idx = []
        train_unlabeled_idx = []
        val_unlabeled_idx = []
        #TODO this is horribly inefficient
        if len(labeled_idx) > 0:
            cell_types = adata[labeled_idx].obs[cell_type_key].unique().tolist()
            for cell_type in cell_types:
                ct_idx = labeled_idx[adata[labeled_idx].obs[cell_type_key] == cell_type]
                n_train_samples = int(np.ceil(train_frac * len(ct_idx)))
                np.random.shuffle(ct_idx)
                train_labeled_idx.append(ct_idx[:n_train_samples])
                val_labeled_idx.append(ct_idx[n_train_samples:])
        if len(unlabeled_idx) > 0:
            n_train_samples = int(np.ceil(train_frac * len(unlabeled_idx)))
            train_unlabeled_idx.append(unlabeled_idx[:n_train_samples])
            val_unlabeled_idx.append(unlabeled_idx[n_train_samples:])
        train_idx = train_labeled_idx + train_unlabeled_idx
        val_idx = val_labeled_idx + val_unlabeled_idx

        train_idx = np.concatenate(train_idx)
        val_idx = np.concatenate(val_idx)

    elif condition_key is not None:
        train_idx = []
        val_idx = []
        conditions = adata.obs[condition_key].unique().tolist()
        for condition in conditions:
            cond_idx = indices[adata.obs[condition_key] == condition]
            n_train_samples = int(np.ceil(train_frac * len(cond_idx)))
            #np.random.shuffle(cond_idx)
            train_idx.append(cond_idx[:n_train_samples])
            val_idx.append(cond_idx[n_train_samples:])

        train_idx = np.concatenate(train_idx)
        val_idx = np.concatenate(val_idx)

    else:
        n_train_samples = int(np.ceil(train_frac * len(indices)))
        np.random.shuffle(indices)
        train_idx = indices[:n_train_samples]
        val_idx = indices[n_train_samples:]

    return train_idx, val_idx


def make_dataset(adata_ccPairs,
                 adata_rest,
                 train_frac=0.9,
                 condition_key=None,
                 cell_type_keys=None,
                 condition_encoder=None,
                 cell_type_encoder=None,
                 labeled_indices=None,
                 ):
    """Splits 'adata' into train and validation data and converts them into 'CustomDatasetFromAdata' objects.

       Parameters
       ----------

       Returns
       -------
       data_set_train_ccPairs: CustomDatasetFromAdata
            Training dataset for ccPairs cells
       data_set_valid_ccPairs: CustomDatasetFromAdata
            Validation dataset for ccPairs cells
       data_set_train_rest: CustomDatasetFromAdata
            Training dataset for rest cells
       data_set_valid_rest: CustomDatasetFromAdata
            Validation dataset for rest cells
    """
    # Preprare data of ccPairs cells for semisupervised learning
    print(f"Preparing {adata_ccPairs.shape} for ccPairs cells")
    labeled_array = np.zeros((len(adata_ccPairs), 1))
    if labeled_indices is not None:
        labeled_array[labeled_indices] = 1

    if cell_type_keys is not None:
        finest_level = None
        n_cts = 0
        for cell_type_key in cell_type_keys:
            if len(adata_ccPairs.obs[cell_type_key].unique().tolist()) >= n_cts:
                n_cts = len(adata_ccPairs.obs[cell_type_key].unique().tolist())
                finest_level = cell_type_key
        print(f"Splitting data {adata_ccPairs.shape} for ccPairs cells")
        train_idx, val_idx = train_test_split(adata_ccPairs, train_frac, cell_type_key=finest_level,
                                              labeled_array=labeled_array)

    elif condition_key is not None:
        train_idx, val_idx = train_test_split(adata_ccPairs, train_frac, condition_key=condition_key)
    else:
        train_idx, val_idx = train_test_split(adata_ccPairs, train_frac)

    print("Instantiating dataset of ccPairs cells")
    data_set_train_ccPairs = ccVAEDataset(
        adata_ccPairs if train_frac == 1 else adata_ccPairs[train_idx],
        adata_rest = None,
        use_ccPairs = 'ccPairs',
        condition_key=condition_key,
        cell_type_keys=cell_type_keys,
        condition_encoder=condition_encoder,
        cell_type_encoder=cell_type_encoder,
        labeled_array=labeled_array[train_idx]
    )
    if train_frac == 1:
        data_set_valid_ccPairs = None
    else:
        data_set_valid_ccPairs = ccVAEDataset(
            adata_ccPairs[val_idx],
            adata_rest = None,
            use_ccPairs = 'ccPairs',
            condition_key=condition_key,
            cell_type_keys=cell_type_keys,
            condition_encoder=condition_encoder,
            cell_type_encoder=cell_type_encoder,
            labeled_array=labeled_array[val_idx]
        )

    # Preprare data of rest cells for semisupervised learning
    print(f"Preparing {adata_rest.shape} for rest cells")
    labeled_array = np.zeros((len(adata_rest), 1))
    if labeled_indices is not None:
        labeled_array[labeled_indices] = 1

    if cell_type_keys is not None:
        finest_level = None
        n_cts = 0
        for cell_type_key in cell_type_keys:
            if len(adata_rest.obs[cell_type_key].unique().tolist()) >= n_cts:
                n_cts = len(adata_rest.obs[cell_type_key].unique().tolist())
                finest_level = cell_type_key
        print(f"Splitting data {adata_rest.shape}")
        train_idx, val_idx = train_test_split(adata_rest, train_frac, cell_type_key=finest_level,
                                              labeled_array=labeled_array)

    elif condition_key is not None:
        train_idx, val_idx = train_test_split(adata_rest, train_frac, condition_key=condition_key)
    else:
        train_idx, val_idx = train_test_split(adata_rest, train_frac)

    print("Instantiating dataset of rest cells")
    data_set_train_rest = ccVAEDataset(
        adata_rest = adata_rest if train_frac == 1 else adata_rest[train_idx],
        adata_ccPairs = None,
        use_ccPairs = 'rest',
        condition_key=condition_key,
        cell_type_keys=cell_type_keys,
        condition_encoder=condition_encoder,
        cell_type_encoder=cell_type_encoder,
        labeled_array=labeled_array[train_idx]
    )
    if train_frac == 1:
        data_set_valid_rest = None
    else:
        data_set_valid_rest = ccVAEDataset(
            adata_rest = adata_rest[val_idx],
            adata_ccPairs = None,
            use_ccPairs = 'rest',
            condition_key=condition_key,
            cell_type_keys=cell_type_keys,
            condition_encoder=condition_encoder,
            cell_type_encoder=cell_type_encoder,
            labeled_array=labeled_array[val_idx]
        )

    return data_set_train_ccPairs, data_set_valid_ccPairs, data_set_train_rest, data_set_valid_rest


def custom_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)

    elif isinstance(elem, container_abcs.Mapping):
        output = {key: custom_collate([d[key] for d in batch]) for key in elem}
        return output
      

