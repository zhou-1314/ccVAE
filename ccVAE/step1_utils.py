import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad 
import random
from functools import reduce
from collections import Counter
from annoy import AnnoyIndex

from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.stats import zscore
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from sklearn.cross_decomposition import CCA


### The function of step 1 derived from ccVAE model 
def normalize(df):
    """Center each column and scale each column to have unit standard deviation.

    Parameters
    ----------
    df : array-like of shape (n_samples, n_features)
        Two dimensional array to be normalized.
    Returns
    -------
    df : array-like of shape (n_samples, n_features)
        Two dimensional array after normalization.
    """
    df = df - np.mean(df, axis=0)
    df = df / np.std(df, axis=0)
    return df


def log_normalize(X, scale_factor=1e6):
    """Log-normalize the data.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Two dimensional array to normalize.
    scale_factor: float, default=1e6
        Multiple all entries of X by scale_factor before taking log.

    Returns
    -------
    X: array-like of shape (n_samples, n_features)
        Two dimensional array after normalization.
    """
    row_cnts = np.sum(X, axis=1)
    X = ((X * scale_factor).T / row_cnts).T
    X = np.log(X + 1)
    return X


def zscore_normalize(matrix):
    # 对矩阵进行 Z-score 标准化
    zscore_matrix = zscore(matrix, axis=0)
    
    return zscore_matrix


def split_adata_by_batch(adata, batch_key):
    # 使用 batch_key 列将 adata 分组
    grouped = adata.obs.groupby(batch_key)

    # 存储子对象的列表
    sub_adatas = []

    # 遍历分组，提取子对象
    for group, indices in grouped.groups.items():
        # 根据索引提取子对象
        sub_adata = adata[indices].copy()

        # 将子对象添加到列表中
        sub_adatas.append(sub_adata)

    return sub_adatas

## 移除  ValueError: array must not contain infs or NaNs
def remove_inf_nan_arr(arr):
    # Replace inf values with NaN
    arr = np.where(np.isinf(arr), np.nan, arr)
    
    # Drop rows with NaN values
    arr = arr[:, ~np.isnan(arr).any(axis=0)]
    
    return arr

### 针对数据框
def remove_inf_nan_df(df):
    # Replace inf values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop cols with NaN values
    df = df.dropna(axis=1)
    
    return df

def remove_zero_std_columns(df):
    """
    Remove columns with zero standard deviation from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame from which columns with zero standard deviation will be removed.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing only the columns with non-zero standard deviation.
    """

    # Calculate the standard deviation for each column
    std_values = df.std()

    # Find columns with non-zero standard deviation
    non_zero_std_columns = std_values[std_values != 0].index

    # Keep only the columns with non-zero standard deviation
    df_filtered = df[non_zero_std_columns]

    return df_filtered


def select_n_components(singular_values, threshold):
    """
    Select the number of components based on singular values.

    Parameters
    ----------
    singular_values : array-like
        Singular values.
    threshold : float
        Threshold value to determine the importance of singular values.

    Returns
    -------
    n_components : int
        Number of components to retain.
    """
    total_variance = sum(singular_values ** 2)
    variance_explained = np.cumsum(singular_values ** 2) / total_variance
    n_components = np.sum(variance_explained < threshold) + 1
    return n_components


def find_hvgs(adata_list, n_top_genes=2000):
    """
    Find highly variable genes (HVGs) across multiple datasets.

    Parameters
    ----------
    adata_list : list of AnnData objects
        List of AnnData objects containing the datasets.
    n_top_genes : int, optional
        Number of top highly variable genes to identify, by default 2000.

    Returns
    -------
    set
        Set of common highly variable genes (HVGs) across all datasets.
    set
        Set of distinct highly variable genes (HVGs) that are specific to each dataset.
    list
        List of sets of highly variable genes (HVGs) for each dataset in the same order as `adata_list`.
    """

    def get_hvgs(adata):
        # Normalize and log-transform the data
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Identify highly variable genes (HVGs)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

        return set(adata.var_names[adata.var['highly_variable']])

    hvgs_list = []
    for adata in adata_list:
        hvgs = get_hvgs(adata)
        hvgs_list.append(hvgs)

    # Find common highly variable genes (HVGs) across all datasets
    common_hvgs = reduce(lambda x, y: x.intersection(y), hvgs_list)

    # Find distinct highly variable genes (HVGs) specific to each dataset
    distinct_hvgs_list = [hvgs.difference(common_hvgs) for hvgs in hvgs_list]
    merged_distinct_hvgs = set().union(*distinct_hvgs_list)

    return common_hvgs, merged_distinct_hvgs, hvgs_list


def partition_adatas(adata, batch_key):
    """
    Extract sub-AnnData objects for each pair of batches in the `adata` object.

    Parameters:
        adata: AnnData object
            An AnnData object containing single-cell data.

        batch_key: str
            The key used to identify the batches in `adata`.

    Returns:
        sub_adatas: list
            A list of sub-AnnData objects corresponding to each pair of batches.
    """
    batches = adata.obs[batch_key].unique()
    sub_adatas = []

    for i in range(len(batches)):
        for j in range(i + 1, len(batches)):
            batch1 = batches[i]
            batch2 = batches[j]

            sub_adata1 = adata[adata.obs[batch_key] == batch1].copy()
            sub_adata2 = adata[adata.obs[batch_key] == batch2].copy()

            sub_adata1.obs[batch_key] = sub_adata1.obs[batch_key].astype(str)
            sub_adata2.obs[batch_key] = sub_adata2.obs[batch_key].astype(str)

            sub_adata = ad.concat([sub_adata1, sub_adata2], axis=0, join="outer")
            sub_adatas.append(sub_adata)

    return sub_adatas
  


def cdist_correlation(X, Y):
    """Calculate pair-wise Pearson correlation between X and Y.

    Parameters
    ----------
    X: array-like of shape (n_samples_of_X, n_features)
        First dataset.
    Y: array-like of shape (n_samples_of_Y, n_features)
        Second dataset.

    Returns
    -------
    array-like of shape (n_samples_of_X, n_samples_of_Y)
        The (i, j)-th entry is the Pearson correlation between i-th row of X and j-th row of Y.
    """
    n, p = X.shape
    m, p2 = Y.shape
    assert p2 == p

    X = (X.T - np.mean(X, axis=1)).T
    Y = (Y.T - np.mean(Y, axis=1)).T

    X = (X.T / np.sqrt(np.sum(X ** 2, axis=1))).T
    Y = (Y.T / np.sqrt(np.sum(Y ** 2, axis=1))).T

    return 1 - X @ Y.T


def check_min_dist(dist_mat, min_dist):
    """Make sure min(dist_mat) >= min_dist.

    Parameters
    ----------
    dist_mat : array-like of shape (n_samples_1, n_samples_2)
        A two-dimensional distance matrix.
    min_dist : flaot
        Desired minimum distance.

    Returns
    -------
    dist_mat : array-like of shape (n_samples_1, n_samples_2)
        A two-dimensional distance matrix with min(dist_mat) >= min_dist.
    """
    current_min_dist = np.min(dist_mat)
    if current_min_dist < min_dist:
        dist_mat = dist_mat - current_min_dist + min_dist

    return dist_mat
  
  

def search_minimum_sparsity(dist_mat, slackness, init_sparsity,
                            m_min, m_max, num_cells_to_use,
                            min_dist, verbose=True):
    """
    Use binary search to search for the minimum sparsity level k such that
    if dist_mat is trimmed to be a k-NN graph, a valid matching still exists.
    The search starts with k_left=1 and k_right=dist_mat.shape[0], and it is always true that:
    1) any k < k_left doesn't give a valid matching;
    2) any k >= k_right gives a valid matching.

    Parameters
    ----------
    dist_mat : array-like of shape (n_samples_1, n_samples_2)
        A two-dimensional distance matrix.
    slackness : int
        Binary search terminates when k_right - k_left <= slackness;
        an exact binary search corresponds to slackness = 0
    init_sparsity : int
        Binary search starts from k=init_sparsity.
    m_min : int
        Each row in the first dataset is matched to at least m_min many rows in the second dataset.
    m_max : int
        Each row in the first dataset is matched to at most m_max many rows in the second dataset.
    num_cells_to_use : int
        Total number of rows to use in the second dataset.
    min_dist : float
        It must be true that minimum entry in dist_mat is >= min_dist.

    Returns
    -------
    k_left : int
        If sparsity<k_left, then there is no valid matching.
    k_right : int
        If sparsity>=k_right, then there is a valid matching.
    """
    assert np.min(dist_mat) >= min_dist
    n1, n2 = dist_mat.shape

    num_sinks = max(n1 * m_max - num_cells_to_use, 0)

    # construct distance matrix that's ready for matching
    if m_max > 1:
        dist_mat = np.tile(dist_mat, (m_max, 1))

    argsort_res = np.argsort(dist_mat, axis=1)

    k_left = 1
    k_right = n2
    cnt = 0

    # start binary search
    while k_left < k_right - slackness:

        if verbose:
            print(
                'If sparsity>={}, then there is a valid matching; '
                'if sparsity<{}, then there is no valid matching.'.format(k_right, k_left), flush=True
            )
        if cnt == 0 and init_sparsity is not None:
            k = init_sparsity
        else:
            k = (k_left + k_right) // 2

        # construct k-NN graph from dist_mat
        # indices for the largest k entries
        largest_k = argsort_res[:, -(n2 - k):]
        # one means there is an edge and zero means there's no edge
        dist_bin = np.ones_like(dist_mat)
        dist_bin[np.arange(dist_mat.shape[0])[:, None], largest_k] = 0

        # add sinks if necessary
        if num_sinks > 0:
            # again, one means there is an edge,
            # and zero means there is no edge (i.e., the distance is np.inf)
            dist_sinks_bin = np.ones((dist_mat.shape[0], num_sinks))
            dist_sinks_bin[:m_min * n1, :] = 0
            dist_bin = np.concatenate((dist_bin, dist_sinks_bin), axis=1)

        dist_bin = csr_matrix(dist_bin)

        col_idx = maximum_bipartite_matching(dist_bin, perm_type='column')
        n_matched = np.sum(col_idx != -1)

        if n_matched == dist_bin.shape[0]:
            # the current k gives a valid matching
            # can trim more aggressively --> try a smaller k
            k_right = k
        else:
            # the current k doesn't give a valid matching
            # try a larger k
            k_left = k + 1

        cnt = cnt + 1

    # we know that k_right must give a valid matching
    if verbose:
        print(
            'If sparsity>={}, then there is a valid matching; '
            'if sparsity<{}, then there is no valid matching.'.format(k_right, k_left), flush=True
        )
    return k_left, k_right


def get_matching_from_indices(col_idx, n1, n2, m_max):
    """
    Assume col_idx is obtained from min_weight_full_bipartite_matching or linear_sum_assignment
    with some dist_mat as input.
    And this dist_mat is organized such that the sinks are in dist_mat[:, :n2].
    This function calculates the matching from col_idx.

    Parameters
    ----------
    col_idx : list
        output from min_weight_full_bipartite_matching or linear_sum_assignment.
    n1 : int
        Sample size of the first dataset.
    n2 : int
        Sample size of the second dataset.
    m_max : int
        Each row in the first dataset is matched to at most m_max many rows in the second dataset.

    Returns
    -------
    list
        A list of (potentially variable length) lists;
        it holds that the i-th row in the first dataset is matched to the res[i]-th row in the second dataset.
    """
    if m_max == 1:
        # make sure the result is a list of (length-one) lists
        return [[col_idx[ii]] for ii in range(n1)]

    res = {ii: [] for ii in range(n1)}

    for kk in range(m_max):
        for ii in range(n1):
            candidate = col_idx[ii + (kk - 1) * n1]
            if candidate < n2:
                res[ii].append(candidate)

    return [res[ii] for ii in range(n1)]


def match_cells(dist_mat, sparsity, m_min, m_max, num_cells_to_use,
                min_dist, mode='auto'):
    """
    Given dist_mat, first trim it to a k-NN graph according to the desired sparsity
    sparsity level, then do a matching according to the specified parameters.

    Parameters
    ----------
    dist_mat : array-like of shape (n_samples_1, n_samples_2)
        A two-dimensional distance matrix.
    sparsity : int
        An integer k such that dist_mat will be trimmed into a k-NN graph.
    m_min : int
        Each row in the first dataset is matched to at least m_min many rows in the second dataset.
    m_max : int
        Each row in the first dataset is matched to at most m_max many rows in the second dataset.
    num_cells_to_use : int
        Total number of samples to use in the second dataset.
    min_dist : float
        It must be true that minimum entry in dist_mat is >= min_dist.
    mode : str, default='auto'
        If 'sparse', use min_weight_full_bipartite_matching;
        if 'dense', use linear_sum_assignment;
        if 'auto': when sparsity<=n//2, use 'sparse', else use 'dense'.

    Returns
    -------
    list
        A list of (potentially variable length) lists;
        it holds that the i-th row in the first dataset is matched to the res[i]-th row in the second dataset.
    """
    assert np.min(dist_mat) >= min_dist
    n1, n2 = dist_mat.shape

    if m_max > 1:
        dist_mat = np.tile(dist_mat, (m_max, 1))

    num_sinks = max(n1 * m_max - num_cells_to_use, 0)

    if sparsity is None:
        mode = 'dense'
    elif mode == 'auto':
        mode = 'sparse' if sparsity <= n2 // 2 else 'dense'

    infinity_placeholder = 0 if mode == 'sparse' else np.Inf

    # trim nodes if necessary
    if sparsity is not None and sparsity != n2:
        argsort_res = np.argsort(dist_mat, axis=1)
        largest_k = argsort_res[:, -(n2 - sparsity):]
        # make a copy because some operations are in-place
        # and may modify the original dist_mat
        dist_mat_cp = np.copy(dist_mat)
        dist_mat_cp[np.arange(dist_mat.shape[0])[:, None], largest_k] = infinity_placeholder
    else:
        dist_mat_cp = dist_mat

    # add sinks if necessary
    if num_sinks > 0:
        # we need make sure that
        # those sinks are favored compared to all other nodes
        dist_sinks = np.zeros((dist_mat.shape[0], num_sinks)) + min_dist / 100
        dist_sinks[:m_min * n1, :] = infinity_placeholder
        dist_mat_cp = np.concatenate((dist_mat_cp, dist_sinks), axis=1)

    if mode == 'sparse':
        dist_mat_cp = csr_matrix(dist_mat_cp)
        _, col_idx = min_weight_full_bipartite_matching(dist_mat_cp)
    elif mode == 'dense':
        _, col_idx = linear_sum_assignment(dist_mat_cp)
    else:
        raise NotImplementedError

    return get_matching_from_indices(col_idx, n1, n2, m_max)
	

def get_cancor(X, Y, n_components=10, max_iter=2000):
    """Fit CCA and calculate the canonical correlations.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features_of_X)
        First dataset.
    Y: array-like of shape (n_samples, n_features_of_Y)
        Second dataset.
    n_components: int, default=2
        Number of CCA components to calculate; must be <= min(n_features_of_X, n_features_of_Y)
    max_iter: int, default=1000
        Maximum number of iterations.

    Returns
    -------
    cancor: array-like of shape (n_components, )
        Vector of canonical components.
    cca: CCA
        CCA object.
    """
    assert X.shape[1] >= n_components
    assert Y.shape[1] >= n_components

    cca = CCA(n_components=n_components, max_iter=max_iter)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    cancor = np.corrcoef(
        X_c, Y_c, rowvar=False).diagonal(
        offset=cca.n_components)
    return cancor, cca


def split_unique_index(unique_index):
    """
    Split the unique index into two separate indexes.

    Parameters
    ----------
    unique_index : pd.MultiIndex
        Unique index to be split.

    Returns
    -------
    pd.Index
        First index.
    pd.Index
        Second index.
    """
    # Split the unique index into two separate indexes
    index1, index2 = zip(*unique_index)

    # Convert to pd.Index objects
    index1 = pd.Index(index1)
    index2 = pd.Index(index2)

    return index1, index2

def extract_subobjects(adata, selected_cells):
    """
    Extract subobjects from the Scanpy object based on the provided cell index and return the complementary subobject.

    Parameters
    ----------
    adata : AnnData
        The original Scanpy object.
    selected_cells : list
        List of cell index to be extracted.

    Returns
    -------
    extracted_adata : AnnData
        Subobject extracted based on the provided cell index.
    complementary_adata : AnnData
        Complementary subobject to the extracted cells.
    """
    # Create a boolean mask to select the cells to be extracted
    mask = [cell in selected_cells for cell in adata.obs_names]

    # Extract the selected cells' data
    extracted_adata = adata[mask].copy()

    # Create a complementary boolean mask to select cells not in the extracted set
    complementary_mask = [not m for m in mask]
    complementary_adata = adata[complementary_mask].copy()

    return extracted_adata, complementary_adata


def split_adata_by_obs(adata, obs_key):
    """
    Split the `adata` object based on the specified observation key (`obs_key`) and return the split data.

    Parameters:
        adata: AnnData object
            An AnnData object containing single-cell data.

        obs_key: str
            The observation key used to split the `adata`.

    Returns:
        df1_sub: pandas DataFrame
            A DataFrame containing the split data for the first subset.

        df2_sub: pandas DataFrame
            A DataFrame containing the split data for the second subset.
    """
    
    # Make sure the `obs_key` provided exists in the `adata` object
    if obs_key not in adata.obs.keys():
        raise ValueError(f"obs_key '{obs_key}' does not exist in adata.obs")

    # Get the metadata column for the specified `obs_key`
    obs_column = adata.obs[obs_key]
    obs_values = np.unique(obs_column)

    # Split the `adata` based on `obs_values`
    split_data_list = []
    split_indices_list = []  # Stores the indices for each split data

    for value in obs_values:
        indices = np.where(obs_column == value)[0]
        split_data = adata[indices].X

        # Convert sparse matrix to dense matrix and set row and column names
        if issparse(split_data):
            split_data = split_data.toarray()
            genes = adata.var_names.tolist()  # Get the list of gene or protein names
            cells = adata.obs_names[indices].tolist()  # Get the list of row names (cell names)
            split_data = pd.DataFrame(split_data, index=cells, columns=genes)  # Create a DataFrame with row and column names
        else:
            genes = adata.var_names.tolist()  # Get the list of gene or protein names
            cells = adata.obs_names[indices].tolist()  # Get the list of row names (cell names)
            split_data = pd.DataFrame(split_data, index=cells, columns=genes)  # Create a DataFrame with row and column names

        split_data_list.append(split_data)
        split_indices_list.append(indices)

    df1_sub = split_data_list[0]
    df2_sub = split_data_list[1]
    
    return pd.DataFrame(df1_sub), pd.DataFrame(df2_sub)


def acquire_pairs(X, Y, k, metric, normalization=True):
    """
    Retrieve nearest neighbor pairs (pairs) from two input matrices X and Y.

    Parameters:
        - X: numpy array representing the first input matrix. Shape: (number of samples, number of features).
        - Y: numpy array representing the second input matrix. Shape: (number of samples, number of features).
        - k: Integer specifying the number of nearest neighbors to retrieve.
        - metric: String specifying the distance metric used for computing nearest neighbors.
          It can be "euclidean" for Euclidean distance, "manhattan" for Manhattan distance,
          or "cosine" for cosine distance.

    Returns:
        List of pairs, where each pair is represented as (x, y), indicating the indices of
        nearest neighbors between X and Y.
    """
    #if normalization:
    #  X = normalize(X)
    #  Y = normalize(Y)
      
    f = X.shape[1]
    t1 = AnnoyIndex(f, metric)
    t2 = AnnoyIndex(f, metric)
    for i in range(len(X)):
        t1.add_item(i, X[i])
    for i in range(len(Y)):
        t2.add_item(i, Y[i])
    t1.build(10)
    t2.build(10)

    mnn_mat = np.bool8(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t2.get_nns_by_vector(item, k) for item in X])
    for i in range(len(sorted_mat)):
        mnn_mat[i,sorted_mat[i]] = True
    _ = np.bool8(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t1.get_nns_by_vector(item, k) for item in Y])
    for i in range(len(sorted_mat)):
        _[sorted_mat[i],i] = True
    mnn_mat = np.logical_and(_, mnn_mat)
    pairs = [(x, y) for x, y in zip(*np.where(mnn_mat>0))]
    return pairs


def create_pairs_dict(pairs):
    """
    Create a dictionary of pairs from a list of pairs.

    Parameters:
        - pairs: List of pairs, where each pair is represented as (x, y), indicating the indices of
          nearest neighbors.

    Returns:
        Dictionary where the keys are the indices from the first set and the values are lists of indices
        from the second set.
    """
    pairs_dict = {}
    for x, y in pairs:
        if x not in pairs_dict.keys():
            pairs_dict[x] = [y]
        else:
            pairs_dict[x].append(y)
    return pairs_dict
  
  
def sub_data_preprocess(adata: sc.AnnData, n_top_genes: int=2000,  min_genes: int=200, min_cells: int=3):
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    return adata


def data_preprocess(adata: sc.AnnData, n_top_genes: int=2000,
                    min_genes: int=200, min_cells: int=3):
    hv_adata = sub_data_preprocess(adata, n_top_genes=n_top_genes, min_genes=min_genes, min_cells=min_cells)
    hv_adata = hv_adata[:, hv_adata.var['highly_variable']]
    return hv_adata

def common_genes(adata1, adata2):
    """
    Find the intersection of gene names between two anndata objects and create new anndata objects containing only the common genes.

    Parameters:
        adata1 (anndata.AnnData): The first input anndata object.
        adata2 (anndata.AnnData): The second input anndata object.

    Returns:
        anndata.AnnData, anndata.AnnData: Two new anndata objects containing only the common genes.
    """
    
    # 判断每个基因是否含有NaN值
    nan_mask1 = np.isnan(adata1.X).any(axis=0)
    nan_mask2 = np.isnan(adata2.X).any(axis=0)
    
    # 提取不含NaN值的基因
    adata1_no_nan = adata1[:, ~nan_mask1]
    adata2_no_nan = adata2[:, ~nan_mask2]
    
    # Get gene names from both adata objects
    genes_adata1 = set(adata1_no_nan.var_names)
    genes_adata2 = set(adata2_no_nan.var_names)
    
    # Find the intersection of gene names
    common_genes = list(genes_adata1.intersection(genes_adata2))
    
    # Filter data for common genes in both adata objects
    adata1_common_genes = adata1[:, common_genes]
    adata2_common_genes = adata2[:, common_genes]
    
    return adata1_common_genes, adata2_common_genes

def calculate_mnn_pairs(dataset, k1=None, k2=None, metric='euclidean', 
                        normalization=True, omics = None, n_top_genes=2000,  
                        min_genes=200, min_cells=3, seed=123):
    """
    Calculate mutual nearest neighbor pairs between two datasets.

    Parameters:
        - dataset: A list of two Scanpy AnnData objects representing the datasets to compare.
        - k1: Integer specifying the number of nearest neighbors to retrieve for dataset A.
          If not provided, it is automatically determined based on the size of the datasets.
        - k2: Integer specifying the number of nearest neighbors to retrieve for dataset B.
          If not provided, it is automatically determined based on the size of the datasets.
        - metric: String specifying the distance metric used for computing nearest neighbors.
          It can be "euclidean" for Euclidean distance, "manhattan" for Manhattan distance,
          or "cosine" for cosine distance.
        - other parameter likes sub_data_preprocess function.

    Returns:
        A pandas MultiIndex object representing the mutual nearest neighbor pairs between the two datasets.
    """
    index_set = set()
    
    if normalization and omics == 'scRNA':
      print('Data preprocess of reference...')
      sample = data_preprocess(dataset[0], n_top_genes, min_genes, min_cells)
      sample = sample.X
      print('Data preprocess of query...')
      query_sample = data_preprocess(dataset[1], n_top_genes, min_genes, min_cells)
      query_sample = query_sample.X
      print('PreProcess Done.')
    elif normalization is not True and omics == 'scRNA':
      print('Perform no normalization for HVGs')
      tmp1 = dataset[0].copy()
      tmp1 = data_preprocess(tmp1, n_top_genes, min_genes, min_cells)
      hvg_mask1 = tmp1.var['highly_variable'] 
      sample = dataset[0][:,tmp1.var_names[hvg_mask1]].X
      
      tmp2 = dataset[1].copy()
      tmp2 = data_preprocess(tmp2, n_top_genes, min_genes, min_cells)
      hvg_mask2 = tmp2.var['highly_variable'] 
      query_sample = dataset[1][:,tmp2.var_names[hvg_mask2]].X
      #sample = normalize(sample)
      #query_sample = normalize(query_sample)
    elif normalization and omics == 'protein':
      dataset[0], dataset[1] = common_genes(dataset[0], dataset[1])
      print('Data preprocess of reference (Z-score normalization)...')
      sample = normalize(remove_inf_nan_arr(dataset[0].X))
      print('Data preprocess of query (Z-score normalization)...')
      query_sample = normalize(remove_inf_nan_arr(dataset[1].X))
      print('PreProcess Done.')
    elif normalization is not True and omics == 'protein':
      dataset[0], dataset[1] = common_genes(dataset[0], dataset[1])
      print('Data preprocess of reference (No normalization)...')
      sample = dataset[0].X
      print('Data preprocess of query (No normalization)...')
      query_sample = dataset[1].X
      print('PreProcess Done, but normaliztion was suggested.')

    if (k1 is None) or (k2 is None):
        k2 = int(min(len(sample), len(query_sample)) / 100)
        k1 = max(int(k2 / 2), 1)

    #print('Calculating Anchor Pairs...')
    anchor_pairs = acquire_pairs(sample, sample, k2, metric)
    #print('Calculating Query Pairs...')
    query_pairs = acquire_pairs(query_sample, query_sample, k2, metric)
    #print('Calculating KNN Pairs...')
    pairs = acquire_pairs(sample, query_sample, k1, metric)
    #print(pairs)
    #print('Calculating Random Walk Pairs...')
    anchor_pairs_dict = create_pairs_dict(anchor_pairs)
    #print('--------')
    #print(anchor_pairs_dict)
    query_pairs_dict = create_pairs_dict(query_pairs)
    pair_plus = []
    for x, y in pairs:
        start = (x, y)
        for i in range(50):
            seed_value = seed + i
            random.seed(seed_value)
            pair_plus.append(start)
            start = (random.choice(anchor_pairs_dict[start[0]]), random.choice(query_pairs_dict[start[1]]))

    datasetA = dataset[1][[y for x, y in pair_plus], :]
    datasetB = dataset[0][[x for x, y in pair_plus], :]
    
    index_set.update(zip(datasetB.obs_names, datasetA.obs_names))
    #print('Done.')

    return pd.MultiIndex.from_tuples(sorted(index_set))
