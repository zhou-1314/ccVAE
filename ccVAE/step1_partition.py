### match.py
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
import os
import math
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import issparse

from tqdm import tqdm 
#from tqdm.notebook import tqdm 

from collections import defaultdict
from . import step1_utils

# import match_utils, utils, embed
# from .cluster import spectral_clustering, jr_kmeans


class Mario_Wrappers(object):
    def __init__(self, df1, df2, omics='scRNA', n_top_genes=2000, normalization=True):
        """Initialize the Mario object.

        Parameters
        ----------
        df1 : array-like of shape (n_samples_1, n_features_1)
            The first dataset.
        df2 : array-like of shape (n_samples_2, n_features_2)
            The second dataset.
        omics : scRNA or protein
        normalization : bool, default=True
            If true, center each column and scale each column to have unit standard deviation.
        """
        # convert df1 and df2 to dataframe if they are not
        if not isinstance(df1, pd.DataFrame):
            df1 = pd.DataFrame(df1)
        if not isinstance(df2, pd.DataFrame):
            df2 = pd.DataFrame(df2)

        self.min_dist = 1e-5
        # parameters related to datasets
        if normalization:
            self.df1 = step1_utils.normalize(df1)
            self.df2 = step1_utils.normalize(df2)
        else:
            self.df1 = df1
            self.df2 = df2
        self.n1, self.p1 = df1.shape
        self.n2, self.p2 = df2.shape
        assert self.n1 <= self.n2
        if omics == 'scRNA':
            adata_list = [sc.AnnData(df1), sc.AnnData(df2)]
            common_hvgs, _, hvgs_list = step1_utils.find_hvgs(adata_list, n_top_genes=n_top_genes)
            self.df1 = df1[list(hvgs_list[0])]
            self.df2 = df2[list(hvgs_list[1])]
            self.ovlp_features = list(common_hvgs)
        else:
            self.ovlp_features = [x for x in self.df1.columns if x in self.df2.columns]

        # hyper-parameters
        self.n_matched_per_cell = None
        self.m_min = None
        self.m_max = None
        self.num_cells_to_use = None
        self.num_sinks = None
        self.sparsity = {'ovlp': None, 'all': None}
        self.n_components = {'ovlp': None, 'all': None}

        # cache some results
        self.dist = {'ovlp': None, 'all': None}
        self.matching = {'ovlp': None, 'all': None, 'wted': None, 'final': None, 'knn': None}
        self.best_wt = None
        self.stacked_svd = {'U': None, 's': None, 'Vh': None}
        self.ovlp_cancor = None
        self.ovlp_scores = {'x': None, 'y': None}

    def compute_dist_ovlp(self, n_components=10):
        """Compute distance matrix based on overlapping features.

        Parameters
        ----------
        n_components : int
            Number of SVD components to keep

        Returns
        -------
        dist_ovlp : array-like of shape (n1, n2)
            The distance matrix based on the overlapping features.
        s : array-like of shape (n_components, )
            Vector of singular values.
        """
        if n_components > len(self.ovlp_features):
            warnings.warn("n_components exceed the number of overlapping features,"
                          " set it to be the number of overlapping features.")
            n_components = len(self.ovlp_features)

        self.n_components['ovlp'] = n_components

        if not (
                self.stacked_svd['U'] is not None and self.stacked_svd['s'] is not None
                and self.stacked_svd['Vh'] is not None and len(self.stacked_svd['s']) >= n_components
        ):
            # Cached results are not valid, do SVD
            arr1 = step1_utils.normalize(self.df1[self.ovlp_features]).to_numpy()
            arr2 = step1_utils.normalize(self.df2[self.ovlp_features]).to_numpy()

            self.stacked_svd['U'], self.stacked_svd['s'], self.stacked_svd['Vh'] = \
                randomized_svd(np.concatenate((arr1, arr2), axis=0), n_components=n_components)
            if n_components == len(self.ovlp_features):
                dist_mat = step1_utils.cdist_correlation(arr1, arr2)
            else:
                svd1 = self.stacked_svd['U'][:self.n1, :] @ np.diag(self.stacked_svd['s']) @ self.stacked_svd['Vh']
                svd2 = self.stacked_svd['U'][self.n1:, :] @ np.diag(self.stacked_svd['s']) @ self.stacked_svd['Vh']
                dist_mat = step1_utils.cdist_correlation(svd1, svd2)
        else:
            # use cached results
            svd1 = self.stacked_svd['U'][:self.n1, :n_components] @ np.diag(self.stacked_svd['s'][:n_components]) \
                   @ self.stacked_svd['Vh'][:n_components, :]
            svd2 = self.stacked_svd['U'][self.n1:, :n_components] @ np.diag(self.stacked_svd['s'][:n_components]) \
                   @ self.stacked_svd['Vh'][:n_components, :]
            dist_mat = step1_utils.cdist_correlation(svd1, svd2)

        # make sure min_dist is at least self.min_dist
        dist_mat = step1_utils.check_min_dist(dist_mat, self.min_dist)
        self.dist['ovlp'] = dist_mat
        return dist_mat, self.stacked_svd['s'][:n_components]

    def search_minimum_sparsity(self, dist_mat, slackness=200, init_sparsity=None, verbose=True):
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
        slackness : int, default=200
            Binary search terminates when k_right - k_left <= slackness;
            an exact binary search corresponds to slackness = 0
        init_sparsity : int, default=None
            Binary search starts from k=init_sparsity. If None, start from the middle.
        verbose : bool, default=True
            If True, print the progress.

        Returns
        -------
        k_left : int
            If sparsity<k_left, then there is no valid matching.
        k_right : int
            If sparsity>=k_right, then there is a valid matching.
        """
        return step1_utils.search_minimum_sparsity(
            dist_mat, slackness, init_sparsity, self.m_min,
            self.m_max, self.num_cells_to_use, self.min_dist, verbose
        )

    def specify_matching_params(self, n_matched_per_cell):
        """Specify how many cells in the second dataset are to be matched with one cell in the first dataset.

        Parameters
        ----------
        n_matched_per_cell : int
            How many cells in the second dataset are to be matched with one cell in the first dataset.
        """
        self.n_matched_per_cell = n_matched_per_cell
        if self.n1 * n_matched_per_cell > self.n2:
            raise ValueError("Not enough cells in Y data!")
        self._specify_matching_params(1, n_matched_per_cell, self.n1 * n_matched_per_cell)

    def _specify_matching_params(self, m_min, m_max, num_cells_to_use):
        """Specify the matching parameters.

        Parameters
        ----------
        m_min : int
            Each row in the first dataset is matched to at least m_min many rows in the second dataset.
        m_max : int
            Each row in the first dataset is matched to at most m_max many rows in the second dataset.
        num_cells_to_use : int
            Total number of rows to use in the second dataset.
        -------
        """
        assert m_min >= 1
        assert self.n1 <= num_cells_to_use
        assert num_cells_to_use <= self.n1 * m_max
        assert num_cells_to_use <= self.n2
        # m_max cannot be too large, otherwise a matching does not exist
        assert m_max <= num_cells_to_use - (self.n1 - 1)

        self.m_min = m_min
        self.m_max = m_max
        self.num_cells_to_use = num_cells_to_use
        self.num_sinks = max(self.n1 * m_max - num_cells_to_use, 0)

    def match_cells(self, dist_mat='ovlp', sparsity=None, mode='auto'):
        """Do cell matching.

        Parameters
        ----------
        dist_mat : str or a user-specified distance matrix, default='ovlp'
            If 'ovlp', then match using the distance matrix computed from overlapping features;
            if 'all', then match using the distance matrix computed from all the features;
            if a user-specified array-like of shape (n1, n2), then match using this distance matrix.
        sparsity : int
            Number of nearest neighbors to keep in the distance matrix.
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
        if isinstance(dist_mat, str):
            if dist_mat not in self.dist or self.dist[dist_mat] is None:
                raise ValueError("Distance not found!")
            self.sparsity[dist_mat] = sparsity
            try:
                self.matching[dist_mat] = step1_utils.match_cells(
                    self.dist[dist_mat], sparsity, self.m_min, self.m_max, self.num_cells_to_use, self.min_dist, mode
                )
            except ValueError:
                # too sparse, find the suitable sparsity level
                warnings.warn(
                    'Current sparsity config '
                    'is too sparse, finding a suitable sparsity level...'
                )
                _, new_sparsity = self.search_minimum_sparsity(
                    self.dist[dist_mat], slackness=200, init_sparsity=self.sparsity[dist_mat] + 1, verbose=True
                )
                warnings.warn('The new sparsity level is {}'.format(new_sparsity))
                self.sparsity[dist_mat] = new_sparsity
                self.matching[dist_mat] = step1_utils.match_cells(
                    self.dist[dist_mat], new_sparsity, self.m_min, self.m_max, self.num_cells_to_use, self.min_dist,
                    mode
                )
            return self.matching[dist_mat]
        else:
            try:
                matching = step1_utils.match_cells(
                    dist_mat, sparsity, self.m_min, self.m_max, self.num_cells_to_use, self.min_dist, mode
                )
            except ValueError:
                # too sparse, find the suitable sparsity level
                warnings.warn(
                    'Current sparsity config '
                    'is too sparse, finding a suitable sparsity level...'
                )
                _, new_sparsity = self.search_minimum_sparsity(
                    self.dist[dist_mat], slackness=200, init_sparsity=sparsity + 1, verbose=True
                )
                warnings.warn('The new sparsity level is {}'.format(new_sparsity))
                matching = step1_utils.match_cells(
                    dist_mat, new_sparsity, self.m_min, self.m_max, self.num_cells_to_use, self.min_dist, mode
                )
            return matching

    def _align_modalities(self, matching):
        """Align df1 so so that cell i in df1 is matched to the "averaged cell" in cells matching[i] in df2.

        Parameters
        ----------
        matching : list
            A list of (potentially variable length) lists;
            it holds that the i-th row in the first dataset is matched to the res[i]-th row in the second dataset.

        Returns
        -------
        X : array-like of shape (n_samples, n_features_1)
            The first dataset.
        Y : array-like of shape (n_samples, n_features_2)
            The second dataset after alignment.
        """
        assert len(matching) == self.n1

        # if cell ii in df1 is filtered out, then matching[ii] is an empty list
        X = []
        Y = []
        for ii in range(self.n1):
            if len(matching[ii]) == 0:
                continue

            X.append(self.df1.iloc[ii, :])
            if len(matching[ii]) == 1:
                Y.append(self.df2.iloc[matching[ii][0]])
            else:
                Y.append(self.df2.iloc[matching[ii]].mean(axis=0))

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    def fit_cca(self, matching, n_components=20, max_iter=3000):
        """Align df1 and df2 using matching, then fit a CCA.

        Parameters
        ----------
        matching : str or list
            Either 'ovlp', 'all', 'wted', 'final', or 'knn',
            or a list of (potentially variable length) lists;
            it holds that the i-th row in the first dataset is matched to the res[i]-th row in the second dataset.
        n_components : int
            Number of components for CCA.
        max_iter : int
            Maximum iteration for CCA.

        Returns
        -------
        cancor: array-like of shape (n_components, )
            Vector of canonical components.
        cca: CCA
            CCA object.
        """
        if n_components > min(self.p1, self.p2):
            warnings.warn('n_components must be <= the dimensions of the two datasets, '
                          'set it to be equal to the minimum of the dimensions of the two datasets')
            n_components = min(self.p1, self.p2)
        if isinstance(matching, str):
            X, Y = self._align_modalities(self.matching[matching])
        else:
            X, Y = self._align_modalities(matching)
        cancor, cca = step1_utils.get_cancor(X, Y, n_components, max_iter)

        return cancor, cca

    def compute_dist_all(self, matching='ovlp', n_components=20, max_iter=5000):
        """Given matching, align df1 and df2, fit a CCA, then use CCA scores to get the distance matrix.

        Parameters
        ----------
        matching : str or list
            Either 'ovlp', meaning that we use the matching obtained from the overlapping features;
            or a list of (potentially variable length) lists, cell i in df1 is matched to cell matching[i] in df2.
        n_components : int
            Number of CCA components.
        max_iter : int
            Max number of iterations when fitting CCA.

        Returns
        -------
        dist_mat : array-like of shape (n1, n2)
            The distance matrix.
        cancor: array-like of shape (n_components, )
            Vector of canonical components.
        """
        if n_components <= 1:
            n_components = 2
            warnings.warn('n_components must be at least 2, '
                          'set it to 2')

        if n_components > min(self.p1, self.p2):
            warnings.warn('n_components must be <= the dimensions of the two datasets, '
                          'set it to be equal to the minimum of the dimensions of the two datasets')
            n_components = min(self.p1, self.p2)

        self.n_components['all'] = n_components

        if isinstance(matching, str) and matching == 'ovlp':
            if self.matching['ovlp'] is None:
                raise ValueError('Initial matching not found!')
            if not (
                    self.ovlp_scores['x'] is not None and self.ovlp_scores['y'] is not None
                    and self.ovlp_scores['x'].shape[1] >= n_components
            ):
                # cached results are not valid, do CCA
                cancor, cca = self.fit_cca(self.matching['ovlp'], n_components, max_iter)
                self.ovlp_cancor = cancor
                self.ovlp_scores['x'], self.ovlp_scores['y'] = cca.transform(self.df1, self.df2)
				# self.ovlp_scores['x'], self.ovlp_scores['y'] = cca.transform(self.df1.values, self.df2.values)
                dist_mat = step1_utils.cdist_correlation(self.ovlp_scores['x'], self.ovlp_scores['y'])
            else:
                # use cached results
                dist_mat = step1_utils.cdist_correlation(self.ovlp_scores['x'][:, :n_components],
                                                   self.ovlp_scores['y'][:, :n_components])
                cancor = self.ovlp_cancor[:n_components]
        else:
            # use user-specified matching
            cancor, cca = self.fit_cca(matching, n_components, max_iter)
            df1_cca, df2_cca = cca.transform(self.df1, self.df2)
            dist_mat = step1_utils.cdist_correlation(df1_cca, df2_cca)

        self.dist['all'] = step1_utils.check_min_dist(dist_mat, self.min_dist)
        return self.dist['all'], cancor

    def interpolate(self, n_wts=10, top_k=10, verbose=True):
        """
        Let wt_vec be an evenly spaced list from 0 to 1 with length n_wts.
        For each wt in wt_vec, do matching on (1-wt)*dist_ovlp + wt*dist_all,
        and select the best wt according to the mean of top_k canonical correlations.

        Parameters
        ----------
        n_wts : int, default=10
            wt_vec is a evenly spaced list from 0 to 1 with length n_wts.
        top_k : int, default=10
            The mean of top_k canonical correlations is taken as the quality measure.
        verbose : bool, default=True
            Print details if True.

        Returns
        -------
        best_wt : float
            The best wt in wt_vec.
        best_matching : list
            The matching corresponds to best_wt.
        """
        wt_vec = np.linspace(0, 1, n_wts)
        max_cancor = float('-inf')
        for ii in range(n_wts):
            if verbose:
                print('Now at iteration {}, wt={}'.format(ii, wt_vec[ii]), flush=True)
            if ii == 0:
                # curr_dist = self.dist['ovlp']
                curr_matching = self.matching['ovlp']
            elif ii == n_wts - 1:
                # curr_dist = self.dist['all']
                curr_matching = self.matching['all']
            else:
                # ii small --> more close to dist_ovlp
                curr_dist = (1 - wt_vec[ii]) * self.dist['ovlp'] + wt_vec[ii] * self.dist['all']
                if self.sparsity['ovlp'] is None or self.sparsity['all'] is None:
                    curr_sparsity = self.sparsity['ovlp'] or self.sparsity['all']
                else:
                    curr_sparsity = max(self.sparsity['ovlp'], self.sparsity['all'])
                try:
                    curr_matching = self.match_cells(curr_dist, curr_sparsity, 'auto')
                except ValueError:
                    # too sparse, find the suitable sparsity level
                    if verbose:
                        print(
                            'Current sparsity config '
                            'is too sparse, finding a suitable sparsity level...', flush=True
                        )
                    _, curr_sparsity = self.search_minimum_sparsity(
                        curr_dist, slackness=200, init_sparsity=curr_sparsity + 1, verbose=verbose
                    )
                    curr_matching = self.match_cells(curr_dist, curr_sparsity, mode='auto')

            # compute median/mean of top_k cancors
            curr_cancor = np.mean(self.fit_cca(curr_matching, n_components=top_k)[0])
            if curr_cancor > max_cancor:
                max_cancor = curr_cancor
                self.matching['wted'] = curr_matching
                self.best_wt = wt_vec[ii]

        return self.best_wt, self.matching['wted']
     
     
## 对全部的样本随机抽样
class BatchSampler:
    """
    Batch-specific Sampler
    Sampled data of each batch is from the same dataset.
    """
    def __init__(self, batch_id, batch_size, drop_last=False):
        """
        Create a BatchSampler object

        Parameters
        ----------
        batch_id : list
            Batch ID for all sample
        batch_size : int
            Batch size for each sampling
        drop_last : bool, optional
            Drop the last samples that do not complete one batch, by default False
        """
        self.batch_id = batch_id
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        num_samples = len(self.batch_id)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        num_batches = num_samples // self.batch_size
        if not self.drop_last and num_samples % self.batch_size != 0:
            num_batches += 1

        for i in range(num_batches):
            batch_start = i * self.batch_size
            batch_end = min(batch_start + self.batch_size, num_samples)
            batch_indices = indices[batch_start:batch_end]

            yield batch_indices

    def __len__(self):
        num_samples = len(self.batch_id)
        num_batches = num_samples // self.batch_size
        if not self.drop_last and num_samples % self.batch_size != 0:
            num_batches += 1
        return num_batches


 
### run resampling
def perform_resampling(adata, batch_key, num_iterations, sample_ratio=0.8, 
                       drop_last=True, omics='scRNA', nHVGs=2000, npcs=30, 
                       singular_ratio=0.9, canonical_ratio=0.9, seed=123,
                       normalization=True, verbose=False):
    """
    Perform resampling on the `adata` object by iterating `num_iterations` times.

    Parameters:
    ----------
    adata: AnnData object
        An AnnData object containing single-cell data.

    batch_key: str
        The batch key used for resampling.

    num_iterations: int
        The number of iterations for resampling.

    sample_ratio: float, optional (default=0.8)
        The ratio of cells to sample from the dataset.

    drop_last: bool, optional (default=True)
        Whether to drop the last batch if the number of cells is not divisible by the sample ratio.

    omics: str, optional (default='scRNA')
        The omics type for matching. Used by the Mario_Wrappers class.

    nHVGs: int, optional (default=2000)
        The number of highly variable genes to consider. Used by the Mario_Wrappers class.

    npcs: int, optional (default=30)
        The number of principal components to use for computing distances. Used by the Mario_Wrappers class.

    singular_ratio: float, optional (default=0.9)
        The ratio of singular values to consider when selecting the number of components. Used by the step1_utils.select_n_components function.

    canonical_ratio: float, optional (default=0.9)
        The ratio of canonical correlations to consider when selecting the number of components. Used by the step1_utils.select_n_components function.

    normalization: bool, optional (default=True)
        A flag indicating whether to perform normalization during MARIO matching.

    verbose: bool, optional (default=False)
        A flag indicating whether to display verbose output during computations.

    Returns:
    ----------
    index_set: pandas MultiIndex
        A sorted pandas MultiIndex containing the unique index values.
    """

    # Create an empty set to store index values
    index_set = set()
    count = 0  # Record the number of times the condition is satisfied

    while count < num_iterations:
    #for count in tqdm(range(num_iterations), desc="Performing resampling", unit="iteration"):
        # Set random seed for each iteration
        seed_value = seed + count  # Generate a unique seed for each iteration
        np.random.seed(seed_value)

        # Create an instance of BatchSampler
        n_cells = int(adata.n_obs * sample_ratio)
        batch_sampler = BatchSampler(batch_id=adata.obs[batch_key], batch_size=n_cells, drop_last=drop_last)

        # Iterate over the batches
        for batch_indices in batch_sampler:
            # Extract sub-objects from adata using the sampled indices
            sub_adata = adata[batch_indices]

            ## Split the data
            df1_sub, df2_sub = step1_utils.split_adata_by_obs(sub_adata, batch_key)
            
            if omics == 'protein':
              df1_sub = step1_utils.remove_inf_nan_df(df1_sub)
              df2_sub = step1_utils.remove_inf_nan_df(df2_sub)
              #print('df1_sub shape is',df1_sub.shape)
              #print('df2_sub shape is',df2_sub.shape)
            elif omics == 'scRNA':
              df1_sub = step1_utils.remove_zero_std_columns(df1_sub)
              df2_sub = step1_utils.remove_zero_std_columns(df2_sub)

            if df1_sub.shape[0] > df2_sub.shape[0]:
                # Swap the dataframes
                df1_sub, df2_sub = df2_sub, df1_sub

            mario = Mario_Wrappers(df1_sub, df2_sub, normalization=normalization, omics=omics, n_top_genes=nHVGs)

            _, singular_values = mario.compute_dist_ovlp(n_components=npcs)

            # Use singular values to select n_components cut-off = 0.9
            n_components = step1_utils.select_n_components(singular_values, singular_ratio)

            # Any value above six looks good; let us choose 10
            _ = mario.compute_dist_ovlp(n_components=n_components)
            # Specify how many cells in df2 should be matched to one cell in df1, we use the standard 1v1 matching
            #num_matched = math.floor(df2_sub.shape[0]/df1_sub.shape[0])
            mario.specify_matching_params(1)

            # [optional] Check the minimum valid sparsity level
            sparsity_value = mario.search_minimum_sparsity(mario.dist['ovlp'], slackness=1, init_sparsity=100, verbose=verbose)
            sparsity = math.ceil(sparsity_value[0] / 10) * 10  ## Round up to the nearest multiple of 10, e.g., 287 ==> 290

            _ = mario.match_cells('ovlp', sparsity=sparsity, mode='auto')

            ### Parameters for refined all feature matching ###
            # Compute distance matrix using all the features
            _, canonical_correlations = mario.compute_dist_all('ovlp', n_components=npcs)

            # Use canonical_correlations values to select n_components cut-off = 0.9
            cor_components = step1_utils.select_n_components(canonical_correlations, canonical_ratio)

            _, canonical_correlations = mario.compute_dist_all('ovlp', n_components=cor_components)
            # Perform the refined matching
            _ = mario.match_cells('all', sparsity=None, mode='auto')

            ### Find the best interpolation of overlapping and all matching
            best_wt, _ = mario.interpolate(n_wts=5, top_k=5, verbose=verbose)

            ### Matching result access
            ## Reformat the MARIO matching results to a dataframe
            df1_rowidx = list(range(len(mario.matching['wted'])))  # Extract the final matching
            filtered_out = [i for i, x in enumerate(mario.matching['wted']) if not x]
            match_final_df1 = [e for e in df1_rowidx if not e in filtered_out]
            match_final_df2 = [item for sublist in mario.matching['wted'] for item in sublist]
            matching_final_df = pd.DataFrame(np.column_stack([match_final_df1, match_final_df2]))

            # Extract the corresponding data using the index dataframe's index
            final_df1_sub = df1_sub.iloc[match_final_df1]
            final_df2_sub = df2_sub.iloc[match_final_df2]

            # Iterate over each dataframe and add the index values to the set
            index_set.update(zip(final_df1_sub.index, final_df2_sub.index))

        count += 1  # Increase the count when the condition is satisfied
        # 添加进度条
        tqdm.write(f"Iteration {count}/{num_iterations}")

    return pd.MultiIndex.from_tuples(sorted(index_set))
  



def run_partition(adata, batch_key, num_iterations=10, sample_ratio=0.8, drop_last=True, omics='scRNA',
                  nHVGs=2000, npcs=30, singular_ratio=0.9, canonical_ratio=0.9, metric='euclidean', k1=20, k2=10,
                  normalization=True, min_genes=200, min_cells=3, seed=123, verbose=False):
    """
    Process multiple batches of data in the `adata` object using MNN correction and cell partitioning.

    Parameters:
    ----------
    adata: AnnData object
        An AnnData object containing single-cell data.

    batch_key: str
        The batch key used for resampling.

    num_iterations: int
        The number of iterations for resampling.

    sample_ratio: float, optional (default=0.8)
        The ratio of cells to sample from the dataset.

    drop_last: bool, optional (default=True)
        Whether to drop the last batch if the number of cells is not divisible by the sample ratio.

    omics: str, optional (default='scRNA')
        The omics type for matching. Used by the Mario_Wrappers class.

    nHVGs: int, optional (default=2000)
        The number of highly variable genes to consider. Used by the Mario_Wrappers class.

    npcs: int, optional (default=30)
        The number of principal components to use for computing distances. Used by the Mario_Wrappers class.

    singular_ratio: float, optional (default=0.9)
        The ratio of singular values to consider when selecting the number of components. Used by the step1_utils.select_n_components function.

    canonical_ratio: float, optional (default=0.9)
        The ratio of canonical correlations to consider when selecting the number of components. Used by the step1_utils.select_n_components function.

    metric: str, optional (default='euclidean')
        String specifying the distance metric used for computing nearest neighbors.
        It can be "euclidean" for Euclidean distance, "manhattan" for Manhattan distance,
        or "cosine" for cosine distance.

    k1: int, optional (default=20)
        Integer specifying the number of nearest neighbors to retrieve for dataset A.
        If not provided, it is automatically determined based on the size of the datasets.
    
    k2: int, optional (default=10)
        Integer specifying the number of nearest neighbors to retrieve for dataset B.
        If not provided, it is automatically determined based on the size of the datasets.

    normalization: bool, optional (default=True)
        A flag indicating whether to perform normalization during MARIO matching.

    other parameters like nHVGs, min_genes, min_cells, int,  optional (nHVGs=2000,  min_genes=200, min_cells=3)
        if normalization=True, these parameter should be set, please refer to preprocess functions of scanpy.
    
    verbose: bool, optional (default=False)
        A flag indicating whether to display verbose output during computations.

    Returns:
    ----------
    ccPairs_adata: AnnData object
        An AnnData object containing the ccPairs cells with batch effect.

    rest_adata: AnnData object
        An AnnData object containing the rest cells without batch effect.
    """
    batch_adatas = step1_utils.partition_adatas(adata, batch_key)
    index_Allbatch = []

    print('Preparing to calculating Anchor and Query Pairs...')
    for batch_adata in batch_adatas:
        ccPairs_index = perform_resampling(batch_adata, batch_key=batch_key, num_iterations=num_iterations,
                                             sample_ratio=sample_ratio, drop_last=drop_last,omics=omics, nHVGs=nHVGs,
                                             npcs=npcs, singular_ratio=singular_ratio, canonical_ratio=canonical_ratio,
                                             normalization=normalization, verbose=verbose)

        index1, index2 = step1_utils.split_unique_index(ccPairs_index)
        ccPairs_adata, _ = step1_utils.extract_subobjects(adata,index1.append(index2))
        adata_list = step1_utils.split_adata_by_batch(ccPairs_adata,batch_key)
        mnn_pairs = step1_utils.calculate_mnn_pairs(adata_list, k1=k1, k2=k2, metric=metric, normalization=normalization,
                                                      omics=omics, n_top_genes=nHVGs, min_genes=min_genes, min_cells=min_cells)
        index_Allbatch.append(mnn_pairs)
          
      print('Finishing calculating cell-cell Pairs...')
  
      merged_index = index_Allbatch[0]
  
      for index in index_Allbatch[1:]:
          merged_index = merged_index.append(index)
  
      index1, index2 = step1_utils.split_unique_index(merged_index)
      ccPairs_adata, rest_adata = step1_utils.extract_subobjects(adata, index1.append(index2))
  
      return ccPairs_adata, rest_adata

