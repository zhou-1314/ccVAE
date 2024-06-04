import torch
import numpy as np
import anndata as ad
import scanpy as sc
from scipy import sparse
from functools import reduce

def one_hot_encoder(idx, n_cls):
    """
    Convert categorical indices to one-hot encoded representation.

    Parameters:
        idx: torch.Tensor
            Tensor containing categorical indices to be converted to one-hot encoding.

        n_cls: int
            Number of classes/categories in the dataset.

    Returns:
        onehot: torch.Tensor
            One-hot encoded representation of the input indices.
    """
    assert torch.max(idx).item() < n_cls
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot


def partition(data, partitions, num_partitions):
    """
    Partition the data based on given partitions.

    Parameters:
        data: list or torch.Tensor
            Data to be partitioned.

        partitions: torch.Tensor
            Tensor containing partition indices for each data point.

        num_partitions: int
            Number of partitions.

    Returns:
        res: list
            List of data partitions.
    """
    res = []
    partitions = partitions.flatten()
    for i in range(num_partitions):
        indices = torch.nonzero((partitions == i), as_tuple=False).squeeze(1)
        res += [data[indices]]
    return res


def remove_sparsity(adata):
    """
        If ``adata.X`` is a sparse matrix, this will convert it in to normal matrix.
        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        Returns
        -------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
    """
    if sparse.issparse(adata.X):
        new_adata = sc.AnnData(X=adata.X.A, obs=adata.obs.copy(deep=True), var=adata.var.copy(deep=True))
        return new_adata

    return adata


def label_encoder(adata, encoder, condition_key=None):
    """Encode labels of Annotated `adata` matrix.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       encoder: Dict
            dictionary of encoded labels.
       condition_key: String
            column name of conditions in `adata.obs` data frame.

       Returns
       -------
       labels: `~numpy.ndarray`
            Array of encoded labels
       label_encoder: Dict
            dictionary with labels and encoded labels as key, value pairs.
    """
    unique_conditions = list(np.unique(adata.obs[condition_key]))
    labels = np.zeros(adata.shape[0])

    if not set(unique_conditions).issubset(set(encoder.keys())):
        missing_labels = set(unique_conditions).difference(set(encoder.keys()))
        print(f"Warning: Labels in adata.obs[{condition_key}] is not a subset of label-encoder!")
        print(f"The missing labels are: {missing_labels}")
        print("Therefore integer value of those labels is set to -1")
        for data_cond in unique_conditions:
            if data_cond not in encoder.keys():
                labels[adata.obs[condition_key] == data_cond] = -1

    for condition, label in encoder.items():
        labels[adata.obs[condition_key] == condition] = label
    return labels


def data_preprocess(adata_ccPairs, adata_rest, n_top_genes=2000, filter_min_counts=True, scale_factor=True, use_scale=True, use_logtrans=True,
                    counts_per_cell=False, select_gene_desc=False, select_gene_adclust=False, use_count=False):
    
    # Filter genes and cells based on minimum counts
    if filter_min_counts:
        if use_count:
            #sc.pp.filter_genes(adata_ccPairs, min_counts=1)
            sc.pp.filter_cells(adata_ccPairs, min_counts=1)
            #sc.pp.filter_genes(adata_rest, min_counts=1)
            sc.pp.filter_cells(adata_rest, min_counts=1)
        else:
            #sc.pp.filter_genes(adata_ccPairs, min_cells=3)
            sc.pp.filter_cells(adata_ccPairs, min_genes=200)
            #sc.pp.filter_genes(adata_rest, min_cells=3)
            sc.pp.filter_cells(adata_rest, min_genes=200)

    # Set raw attribute of AnnData objects
    if scale_factor or use_scale or use_logtrans:
        adata_ccPairs.raw = adata_ccPairs.copy()
        adata_rest.raw = adata_rest.copy()
    else:
        adata_ccPairs.raw = adata_ccPairs
        adata_rest.raw = adata_rest

    # Normalize per cell and calculate scale factor
    if scale_factor:
        sc.pp.normalize_per_cell(adata_ccPairs)
        adata_ccPairs.obs['scale_factor'] = adata_ccPairs.obs.n_counts / adata_ccPairs.obs.n_counts.median()
        sc.pp.normalize_per_cell(adata_rest)
        adata_rest.obs['scale_factor'] = adata_rest.obs.n_counts / adata_rest.obs.n_counts.median()
    else:
        adata_ccPairs.obs['scale_factor'] = 1.0
        adata_rest.obs['scale_factor'] = 1.0

    # Normalize counts per cell
    if counts_per_cell:
        sc.pp.normalize_per_cell(adata_ccPairs, counts_per_cell_after=1e4)
        sc.pp.normalize_per_cell(adata_rest, counts_per_cell_after=1e4)

    # Perform logarithmic transformation
    if use_logtrans:
        sc.pp.log1p(adata_ccPairs)
        sc.pp.log1p(adata_rest)

    # Select highly variable genes based on mean and dispersion
    if select_gene_desc:
        sc.pp.highly_variable_genes(adata_ccPairs, min_mean=0.0125, max_mean=3, min_disp=0.5, subset=True)
        #adata_ccPairs = adata_ccPairs[:, adata_ccPairs.var['highly_variable']]
        sc.pp.highly_variable_genes(adata_rest, min_mean=0.0125, max_mean=3, min_disp=0.5, subset=True)
        #adata_rest = adata_rest[:, adata_rest.var['highly_variable']]

    # Select highly variable genes using Adclust
    if select_gene_adclust:
        sc.pp.highly_variable_genes(adata_ccPairs, min_mean=None, max_mean=None, min_disp=None, n_top_genes=n_top_genes)
        #adata_ccPairs = adata_ccPairs[:, adata_ccPairs.var.highly_variable]
        sc.pp.highly_variable_genes(adata_rest, min_mean=None, max_mean=None, min_disp=None, n_top_genes=n_top_genes)
        #adata_rest = adata_rest[:, adata_rest.var.highly_variable]

    # Scale the data
    if use_scale:
        sc.pp.scale(adata_ccPairs)
        sc.pp.scale(adata_rest)

    return adata_ccPairs, adata_rest

def merge_subsets(lists):
    # 初始化结果为第一个列表
    merged = set(lists[0])

    # 遍历剩余的列表
    for sublist in lists[1:]:
        merged |= set(sublist)  # 或 merged = merged.union(set(sublist))

    return list(merged)
  

def prepare_adata(adata_ccPairs, adata_rest, n_top_genes=2000, condition_key='batch', filter_min_counts=True,
                    scale_factor=True, use_scale=False, use_logtrans=True,
                    counts_per_cell=True, select_gene_adclust=True,
                    select_gene_desc=False, use_count=False):
    """
    Prepare the data by performing preprocessing steps on the AnnData objects.

    Parameters:
    ----------
    adata_ccPairs: AnnData object
        AnnData object containing scRNA-seq data for ccPairs.

    adata_rest: AnnData object
        AnnData object containing scRNA-seq data for the rest of the cells.
        
    n_top_genes: int, optional (default=2000)
        Number of highly variable genes. default=2000.
        
    condition_key: string, optional (default='batch')
        column name of conditions in `adata.obs` data frame.

    filter_min_counts: bool, optional (default=True)
        Flag indicating whether to filter genes and cells based on minimum counts.

    scale_factor: bool, optional (default=True)
        Flag indicating whether to calculate and apply scale factor normalization.

    use_normalize: bool, optional (default=True)
        Flag indicating whether to perform data normalization.

    use_logtrans: bool, optional (default=True)
        Flag indicating whether to perform logarithmic transformation.

    counts_per_cell: bool, optional (default=False)
        Flag indicating whether to normalize counts per cell.

    select_gene_desc: bool, optional (default=False)
        Flag indicating whether to select highly variable genes based on mean and dispersion.

    select_gene_adclust: bool, optional (default=False)
        Flag indicating whether to select highly variable genes using Adclust.

    use_count: bool, optional (default=False)
        Flag indicating whether to use raw counts for filtering genes and cells.

    Returns:
    ----------
    adata_ccPairs: AnnData object
        Processed AnnData object for ccPairs.

    adata_rest: AnnData object
        Processed AnnData object for the rest of the cells.
    """
    adata_ccPairs, adata_rest = data_preprocess(adata_ccPairs, adata_rest, n_top_genes=n_top_genes, 
                                                filter_min_counts=filter_min_counts,
                                                scale_factor=scale_factor, use_scale=use_scale, 
                                                use_logtrans=use_logtrans,
                                                counts_per_cell=counts_per_cell, 
                                                select_gene_adclust=select_gene_adclust, 
                                                select_gene_desc=select_gene_desc, 
                                                use_count=use_count)
                                                
    hvgs_list = [adata_ccPairs.var_names[adata_ccPairs.var['highly_variable']],
                 adata_rest.var_names[adata_rest.var['highly_variable']]]
    hvgs = merge_subsets(hvgs_list)
    
    try:
      source_adata = adata_ccPairs[:, hvgs]
      target_adata = adata_rest[:, hvgs]
    except KeyError as e:
      invalid_genes = e.args[0].replace("Values ", "").replace(", from", "").replace(" are not valid obs/ var names or indices.", "")
      invalid_genes = invalid_genes.split(", ")
      print(f"The following genes are invalid: {invalid_genes}")
      print("Using the intersection of genes from adata_ccPairs and adata_rest instead.")
      valid_genes = [adata_ccPairs.var_names[adata_ccPairs.var['highly_variable']] & adata_rest.var_names[adata_rest.var['highly_variable']]]
      source_adata = adata_ccPairs[:, valid_genes]
      target_adata = adata_rest[:, valid_genes]
      
    #hvgs_list = [set(adata_ccPairs.var_names[adata_ccPairs.var['highly_variable']]),
    #             set(adata_rest.var_names[adata_rest.var['highly_variable']])]
    #hvgs = merge_subsets(hvgs_list)
    
    #source_adata = adata_ccPairs[:, hvgs]
    source_adata = remove_sparsity(source_adata)
    source_adata.X = np.float32(np.int32(source_adata.X))
    source_conditions = source_adata.obs[condition_key].unique().tolist()

    #target_adata = adata_rest[:, hvgs]
    target_adata = remove_sparsity(target_adata)
    target_adata.X = np.float32(np.int32(target_adata.X))
    target_conditions = target_adata.obs[condition_key].unique().tolist()
    
    return source_adata, target_adata, source_conditions


def prepare_protein_adata(source_adata, target_adata, condition_key='batch', overlap=True):
    """
    Find common variable genes (CVGs) across two datasets.

    Parameters:
    ----------
    source_adata: AnnData object
        An AnnData object containing the source dataset.

    target_adata: AnnData object
        An AnnData object containing the target dataset.
        
    condition_key: string, optional (default='batch')
        Column name of conditions in `adata.obs` data frame.

    overlap: bool, optional (default=True)
        If True, both datasets will be filtered to keep only the common CVGs.
        If False, the datasets will remain unchanged.

    Returns:
    ----------
    source_adata: AnnData object
        An AnnData object containing the source dataset with common CVGs.

    target_adata: AnnData object
        An AnnData object containing the target dataset with common CVGs (or unchanged if overlap=False).
    """
    hvgs_list = [source_adata.var_names, target_adata.var_names]
    common_hvgs = reduce(lambda x, y: x.intersection(y), hvgs_list)

    if overlap:
      # Find columns with any NA values in adata1.X
      na_columns_adata1 = np.isnan(source_adata.X).any(axis=0)
  
      # Find columns with any NA values in adata2.X
      na_columns_adata2 = np.isnan(target_adata.X).any(axis=0)
      
      # Remove columns with any NA values from adata1.X
      source_adata = source_adata[:, ~na_columns_adata1]
  
      # Remove columns with any NA values from adata2.X
      target_adata = target_adata[:, ~na_columns_adata2]
      
      # Get the gene names for adata1 and adata2
      gene_names1 = source_adata.var_names.tolist()
      gene_names2 = target_adata.var_names.tolist()

      # Take the intersection of gene names
      common_genes = list(set(gene_names1).intersection(gene_names2))
      
      source_adata = source_adata[:, common_genes]
      target_adata = target_adata[:, common_genes]
    else:
      source_adata = source_adata[:, common_hvgs]
      target_adata = target_adata[:, common_hvgs]

    source_adata = remove_sparsity(source_adata)
    source_adata.X = np.float32(np.int32(source_adata.X))
    source_conditions = source_adata.obs[condition_key].unique().tolist()

    target_adata = remove_sparsity(target_adata)
    target_adata.X = np.float32(np.int32(target_adata.X))

    return source_adata, target_adata, source_conditions
