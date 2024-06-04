from collections import Counter

import anndata as ad
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import scanpy as sc

from .step2_ccVAE_utils import label_encoder, remove_sparsity



class ccVAEDataset(Dataset):
    """Dataset handler for TRVAE model and trainer.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       condition_key: String
            column name of conditions in `adata.obs` data frame.
       condition_encoder: Dict
            dictionary of encoded conditions.
       cell_type_keys: List
            List of column names of different celltype hierarchies in `adata.obs` data frame.
       cell_type_encoder: Dict
            dictionary of encoded celltypes.
    """
    def __init__(self,
                 adata_ccPairs,
                 adata_rest,
                 use_ccPairs='ccPairs',
                 condition_key=None,
                 condition_encoder=None,
                 cell_type_keys=None,
                 cell_type_encoder=None,
                 labeled_array=None
                 ):
        assert use_ccPairs in ['ccPairs', 'rest'], "use_ccPairs must be 'ccPairs', or 'rest'"
        self.use_ccPairs = use_ccPairs
        self.condition_key = condition_key
        self.condition_encoder = condition_encoder
        self.cell_type_keys = cell_type_keys
        self.cell_type_encoder = cell_type_encoder

        if self.use_ccPairs == 'ccPairs':
            self._is_sparse = sparse.issparse(adata_ccPairs.X)
            self.data_ccPairs = adata_ccPairs.X if self._is_sparse else torch.tensor(adata_ccPairs.X)
            
            size_factors = np.ravel(adata_ccPairs.X.sum(1))

            self.size_factors = torch.tensor(size_factors)

            labeled_array = np.zeros((len(adata_ccPairs), 1)) if labeled_array is None else labeled_array
            self.labeled_vector = torch.tensor(labeled_array)

            # Encode condition strings to integer
            if self.condition_key is not None:
                self.conditions = label_encoder(
                    adata_ccPairs,
                    encoder=self.condition_encoder,
                    condition_key=condition_key,
                )
                self.conditions = torch.tensor(self.conditions, dtype=torch.long)

            # Encode cell type strings to integer
            if self.cell_type_keys is not None:
                self.cell_types = list()
                for cell_type_key in cell_type_keys:
                    level_cell_types = label_encoder(
                        adata_ccPairs,
                        encoder=self.cell_type_encoder,
                        condition_key=cell_type_key,
                    )
                    self.cell_types.append(level_cell_types)

                self.cell_types = np.stack(self.cell_types).T
                self.cell_types = torch.tensor(self.cell_types, dtype=torch.long)
        else:
            self._is_sparse = sparse.issparse(adata_rest.X)
            self.data_rest = adata_rest.X if self._is_sparse else torch.tensor(adata_rest.X)

            size_factors = np.ravel(adata_rest.X.sum(1))

            self.size_factors = torch.tensor(size_factors)

            labeled_array = np.zeros((len(adata_rest), 1)) if labeled_array is None else labeled_array
            self.labeled_vector = torch.tensor(labeled_array)

            # Encode condition strings to integer
            if self.condition_key is not None:
                self.conditions = label_encoder(
                    adata_rest,
                    encoder=self.condition_encoder,
                    condition_key=condition_key,
                )
                self.conditions = torch.tensor(self.conditions, dtype=torch.long)

            # Encode cell type strings to integer
            if self.cell_type_keys is not None:
                self.cell_types = list()
                for cell_type_key in cell_type_keys:
                    level_cell_types = label_encoder(
                        adata_rest,
                        encoder=self.cell_type_encoder,
                        condition_key=cell_type_key,
                    )
                    self.cell_types.append(level_cell_types)

                self.cell_types = np.stack(self.cell_types).T
                self.cell_types = torch.tensor(self.cell_types, dtype=torch.long)

    def __getitem__(self, index):
        if self.use_ccPairs == 'ccPairs':
            outputs = dict()

            if self._is_sparse:
                x = torch.tensor(np.squeeze(self.data_ccPairs[index].toarray()))
            else:
                x = self.data_ccPairs[index]
            outputs["x_ccPairs"] = x

            outputs["labeled_ccPairs"] = self.labeled_vector[index]
            outputs["sizefactor_ccPairs"] = self.size_factors[index]

            if self.condition_key:
                outputs["batch_ccPairs"] = self.conditions[index]

            if self.cell_type_keys:
                outputs["celltypes_ccPairs"] = self.cell_types[index, :]

            return outputs
        else:
            outputs = dict()

            if self._is_sparse:
                x = torch.tensor(np.squeeze(self.data_rest[index].toarray()))
            else:
                x = self.data_rest[index]
            outputs["x_rest"] = x

            outputs["labeled_rest"] = self.labeled_vector[index]
            outputs["sizefactor_rest"] = self.size_factors[index]

            if self.condition_key:
                outputs["batch_rest"] = self.conditions[index]

            if self.cell_type_keys:
                outputs["celltypes_rest"] = self.cell_types[index, :]

            return outputs

    def __len__(self):
        if self.use_ccPairs == 'ccPairs':
            return self.data_ccPairs.shape[0]
        else:
            return self.data_rest.shape[0]

    @property
    def condition_label_encoder(self) -> dict:
        return self.condition_encoder

    @condition_label_encoder.setter
    def condition_label_encoder(self, value: dict):
        if value is not None:
            self.condition_encoder = value

    @property
    def cell_type_label_encoder(self) -> dict:
        return self.cell_type_encoder

    @cell_type_label_encoder.setter
    def cell_type_label_encoder(self, value: dict):
        if value is not None:
            self.cell_type_encoder = value

    @property # 返回一组权重，用于在分层采样中平衡每个条件的样本数量
    def stratifier_weights(self):
        conditions = self.conditions.detach().cpu().numpy()
        condition_coeff = 1. / len(conditions)

        condition2count = Counter(conditions)
        counts = np.array([condition2count[cond] for cond in conditions])
        weights = condition_coeff / counts
        return weights.astype(float)
      
