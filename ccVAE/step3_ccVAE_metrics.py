import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

import scib
from scipy.stats import entropy
import scipy.special
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from .step2_ccVAE_utils import remove_sparsity



### batch correction
def graph_connectivity(adata, label_key='batch'):
    """Graph Connectivity

    Quantify the connectivity of the subgraph per cell type label.
    The final score is the average for all cell type labels :math:`C`, according to the equation:

    .. math::

        GC = \\frac {1} {|C|} \\sum_{c \\in C} \\frac {|{LCC(subgraph_c)}|} {|c|}

    where :math:`|LCC(subgraph_c)|` stands for all cells in the largest connected component and :math:`|c|` stands for all cells of
    cell type :math:`c`.

    :param adata: integrated adata with computed neighborhood graph
    :param label_key: name in adata.obs containing the cell identity labels

    This function can be applied to all integration output types.
    The integrated object (``adata``) needs to have a kNN graph based on the integration output.
    See :ref:`preprocessing` for more information on preprocessing.

    **Examples**

    .. code-block:: python

        # feature output
        scib.pp.reduce_data(
            adata, n_top_genes=2000, batch_key="batch", pca=True, neighbors=True
        )
        scib.me.graph_connectivity(adata, label_key="celltype")

        # embedding output
        sc.pp.neighbors(adata, use_rep="X_emb")
        scib.me.graph_connectivity(adata, label_key="celltype")

        # knn output
        scib.me.graph_connectivity(adata, label_key="celltype")

    """
    if "neighbors" not in adata.uns:
        raise KeyError(
            "Please compute the neighborhood graph before running this function!"
        )

    clust_res = []

    for label in adata.obs[label_key].cat.categories:
        adata_sub = adata[adata.obs[label_key].isin([label])]
        _, labels = connected_components(
            adata_sub.obsp["connectivities"], connection="strong"
        )
        tab = pd.value_counts(labels)
        clust_res.append(tab.max() / sum(tab))

    return np.mean(clust_res)
  
  

def entropy_batch_mixing(adata, label_key='batch',
                         n_neighbors=50, n_pools=50, n_samples_per_pool=100):
    """Computes Entory of Batch mixing metric for ``adata`` given the batch column name.
        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        label_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame.
        n_neighbors: int
            Number of nearest neighbors.
        n_pools: int
            Number of EBM computation which will be averaged.
        n_samples_per_pool: int
            Number of samples to be used in each pool of execution.
        Returns
        -------
        score: float
            EBM score. A float between zero and one.
    """
    adata = remove_sparsity(adata)
    n_cat = len(adata.obs[label_key].unique().tolist())
    print(f'Calculating EBM with n_cat = {n_cat}')

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.X)
    indices = neighbors.kneighbors(adata.X, return_distance=False)[:, 1:]
    batch_indices = np.vectorize(lambda i: adata.obs[label_key].values[i])(indices)

    entropies = np.apply_along_axis(__entropy_from_indices, axis=1, arr=batch_indices, n_cat=n_cat)

    # average n_pools entropy results where each result is an average of n_samples_per_pool random samples.
    if n_pools == 1:
        score = np.mean(entropies)
    else:
        score = np.mean([
            np.mean(entropies[np.random.choice(len(entropies), size=n_samples_per_pool)])
            for _ in range(n_pools)
        ])

    return score


def nmi(adata, label_key, verbose=False, nmi_method='arithmetic'):
    cluster_key = 'cluster'
    opt_louvain(adata, label_key=label_key, cluster_key=cluster_key, function=nmi_helper,
                plot=False, verbose=verbose, inplace=True)

    print('NMI...')
    nmi_score = nmi_helper(adata, group1=cluster_key, group2=label_key, method=nmi_method)

    return nmi_score


def ari(adata, cluster_key, label_key, implementation=None):
    """Adjusted Rand Index

    The adjusted rand index is a chance-adjusted rand index, which evaluates the pair-wise accuracy of clustering vs.
    ground truth label assignments.
    The score ranges between 0 and 1 with larger values indicating better conservation of the data-driven cell identity
    discovery after integration compared to annotated labels.

    :param adata: anndata object with cluster assignments in ``adata.obs[cluster_key]``
    :param cluster_key: string of column in adata.obs containing cluster assignments
    :param label_key: string of column in adata.obs containing labels
    :param implementation: if set to 'sklearn', uses sklearn's implementation,
        otherwise native implementation is taken

    This function can be applied to all integration output types.
    The ``adata`` must contain cluster assignments that are based off the knn graph given or derived from the integration
    method output.
    For this metric you need to include all steps that are needed for clustering.
    See :ref:`preprocessing` for more information on preprocessing.

    **Examples**

    .. code-block:: python

        # feature output
        scib.pp.reduce_data(
            adata, n_top_genes=2000, batch_key="batch", pca=True, neighbors=True
        )
        scib.me.cluster_optimal_resolution(adata, cluster_key="cluster", label_key="celltype")
        scib.me.ari(adata, cluster_key="cluster", label_key="celltype")

        # embedding output
        sc.pp.neighbors(adata, use_rep="X_emb")
        scib.me.cluster_optimal_resolution(adata, cluster_key="cluster", label_key="celltype")
        scib.me.ari(adata, cluster_key="cluster", label_key="celltype")

        # knn output
        scib.me.cluster_optimal_resolution(adata, cluster_key="cluster", label_key="celltype")
        scib.me.ari(adata, cluster_key="cluster", label_key="celltype")

    """
    ## find cluster_optimal_resolution
    scib.me.cluster_optimal_resolution(adata, cluster_key, label_key)
    
    cluster_key = adata.obs[cluster_key].to_numpy()
    label_key = adata.obs[label_key].to_numpy()

    if len(cluster_key) != len(label_key):
        raise ValueError(
            f"different lengths in cluster_key ({len(cluster_key)}) and label_key ({len(label_key)})"
        )

    if implementation == "sklearn":
        return adjusted_rand_score(cluster_key, label_key)

    def binom_sum(x, k=2):
        return scipy.special.binom(x, k).sum()

    n = len(cluster_key)
    contingency = pd.crosstab(cluster_key, label_key)

    ai_sum = binom_sum(contingency.sum(axis=0))
    bi_sum = binom_sum(contingency.sum(axis=1))

    index = binom_sum(np.ravel(contingency))
    expected_index = ai_sum * bi_sum / binom_sum(n, 2)
    max_index = 0.5 * (ai_sum + bi_sum)

    return (index - expected_index) / (max_index - expected_index)
  

def asw(adata, label_key, batch_key):
    print('silhouette score...')
    sil_global = silhouette(adata, group_key=label_key, metric='euclidean')
    _, sil_clus = silhouette_batch(adata, batch_key=batch_key, group_key=label_key,
                                   metric='euclidean', verbose=False)
    sil_clus = sil_clus['silhouette_score'].mean()
    return sil_clus, sil_global


def knn_purity(adata, label_key, n_neighbors=30):
    """Computes KNN Purity metric for ``adata`` given the batch column name.
        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        label_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame.
        n_neighbors: int
            Number of nearest neighbors.
        Returns
        -------
        score: float
            KNN purity score. A float between 0 and 1.
    """
    adata = remove_sparsity(adata)
    labels = LabelEncoder().fit_transform(adata.obs[label_key].to_numpy())

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.X)
    indices = nbrs.kneighbors(adata.X, return_distance=False)[:, 1:]
    neighbors_labels = np.vectorize(lambda i: labels[i])(indices)

    # pre cell purity scores
    scores = ((neighbors_labels - labels.reshape(-1, 1)) == 0).mean(axis=1)
    res = [
        np.mean(scores[labels == i]) for i in np.unique(labels)
    ]  # per cell-type purity

    return np.mean(res)


def __entropy_from_indices(indices, n_cat):
    return entropy(np.unique(indices, return_counts=True)[1].astype(np.int32), base=n_cat)


def nmi_helper(adata, group1, group2, method="arithmetic"):
    """
       This NMI function was taken from scIB:
       Title: scIB
       Authors: Malte Luecken,
                Maren Buettner,
                Daniel Strobl,
                Michaela Mueller
       Date: 4th October 2020
       Code version: 0.2.0
       Availability: https://github.com/theislab/scib/blob/master/scIB/metrics.py

       Normalized mutual information NMI based on 2 different cluster assignments `group1` and `group2`
       params:
        adata: Anndata object
        group1: column name of `adata.obs` or group assignment
        group2: column name of `adata.obs` or group assignment
        method: NMI implementation
            'max': scikit method with `average_method='max'`
            'min': scikit method with `average_method='min'`
            'geometric': scikit method with `average_method='geometric'`
            'arithmetic': scikit method with `average_method='arithmetic'`

       return:
        normalized mutual information (NMI)
    """
    adata = remove_sparsity(adata)

    if isinstance(group1, str):
        group1 = adata.obs[group1].tolist()
    elif isinstance(group1, pd.Series):
        group1 = group1.tolist()

    labels = adata.obs[group2].values
    labels_encoded = LabelEncoder().fit_transform(labels)
    group2 = labels_encoded

    if len(group1) != len(group2):
        raise ValueError(f'different lengths in group1 ({len(group1)}) and group2 ({len(group2)})')

    # choose method
    if method in ['max', 'min', 'geometric', 'arithmetic']:
        nmi_value = normalized_mutual_info_score(group1, group2, average_method=method)
    else:
        raise ValueError(f"Method {method} not valid")

    return nmi_value


def silhouette(adata, group_key, metric='euclidean', scale=True):
    """
       This ASW function was taken from scIB:
       Title: scIB
       Authors: Malte Luecken,
                Maren Buettner,
                Daniel Strobl,
                Michaela Mueller
       Date: 4th October 2020
       Code version: 0.2.0
       Availability: https://github.com/theislab/scib/blob/master/scIB/metrics.py

       wrapper for sklearn silhouette function values range from [-1, 1] with 1 being an ideal fit, 0 indicating
       overlapping clusters and -1 indicating misclassified cells
    """
    adata = remove_sparsity(adata)
    labels = adata.obs[group_key].values
    labels_encoded = LabelEncoder().fit_transform(labels)
    asw = silhouette_score(adata.X, labels_encoded, metric=metric)
    if scale:
        asw = (asw + 1)/2
    return asw


def silhouette_batch(adata, batch_key, group_key, metric='euclidean', verbose=True, scale=True):
    """
       This ASW function was taken from scIB:
       Title: scIB
       Authors: Malte Luecken,
                Maren Buettner,
                Daniel Strobl,
                Michaela Mueller
       Date: 4th October 2020
       Code version: 0.2.0
       Availability: https://github.com/theislab/scib/blob/master/scIB/metrics.py

       Silhouette score of batch labels subsetted for each group.

       params:
        batch_key: batches to be compared against
        group_key: group labels to be subsetted by e.g. cell type
        metric: see sklearn silhouette score
        embed: name of column in adata.obsm

       returns:
        all scores: absolute silhouette scores per group label
        group means: if `mean=True`
    """
    adata = remove_sparsity(adata)
    glob_batches = adata.obs[batch_key].values
    batch_enc = LabelEncoder()
    batch_enc.fit(glob_batches)
    sil_all = pd.DataFrame(columns=['group', 'silhouette_score'])

    for group in adata.obs[group_key].unique():
        adata_group = adata[adata.obs[group_key] == group]
        if adata_group.obs[batch_key].nunique() == 1:
            continue
        batches = batch_enc.transform(adata_group.obs[batch_key])
        sil_per_group = silhouette_samples(adata_group.X, batches, metric=metric)
        # take only absolute value
        sil_per_group = [abs(i) for i in sil_per_group]
        if scale:
            # scale s.t. highest number is optimal
            sil_per_group = [1 - i for i in sil_per_group]
        d = pd.DataFrame({'group': [group] * len(sil_per_group), 'silhouette_score': sil_per_group})
        # sil_all = sil_all.append(d)
        sil_all = pd.concat([sil_all, d], ignore_index=True)
    sil_all = sil_all.reset_index(drop=True)
    sil_means = sil_all.groupby('group').mean()

    if verbose:
        print(f'mean silhouette per cell: {sil_means}')
    return sil_all, sil_means



def opt_louvain(adata, label_key, cluster_key, function=None, resolutions=None,
                inplace=True, plot=False, verbose=True, **kwargs):
    """
       This Louvain Clustering method was taken from scIB:
       Title: scIB
       Authors: Malte Luecken,
                Maren Buettner,
                Daniel Strobl,
                Michaela Mueller
       Date: 4th October 2020
       Code version: 0.2.0
       Availability: https://github.com/theislab/scib/blob/master/scIB/clustering.py

    params:
        label_key: name of column in adata.obs containing biological labels to be
            optimised against
        cluster_key: name of column to be added to adata.obs during clustering.
            Will be overwritten if exists and `force=True`
        function: function that computes the cost to be optimised over. Must take as
            arguments (adata, group1, group2, **kwargs) and returns a number for maximising
        resolutions: list if resolutions to be optimised over. If `resolutions=None`,
            default resolutions of 20 values ranging between 0.1 and 2 will be used
    returns:
        res_max: resolution of maximum score
        score_max: maximum score
        score_all: `pd.DataFrame` containing all scores at resolutions. Can be used to plot the score profile.
        clustering: only if `inplace=False`, return cluster assignment as `pd.Series`
        plot: if `plot=True` plot the score profile over resolution
    """
    adata = remove_sparsity(adata)

    if resolutions is None:
        n = 20
        resolutions = [2 * x / n for x in range(1, n + 1)]

    score_max = 0
    res_max = resolutions[0]
    clustering = None
    score_all = []

    # maren's edit - recompute neighbors if not existing
    try:
        adata.uns['neighbors']
    except KeyError:
        if verbose:
            print('computing neigbours for opt_cluster')
        sc.pp.neighbors(adata)

    for res in resolutions:
        sc.tl.louvain(adata, resolution=res, key_added=cluster_key)
        score = function(adata, label_key, cluster_key, **kwargs)
        score_all.append(score)
        if score_max < score:
            score_max = score
            res_max = res
            clustering = adata.obs[cluster_key]
        del adata.obs[cluster_key]

    if verbose:
        print(f'optimised clustering against {label_key}')
        print(f'optimal cluster resolution: {res_max}')
        print(f'optimal score: {score_max}')

    score_all = pd.DataFrame(zip(resolutions, score_all), columns=('resolution', 'score'))
    if plot:
        # score vs. resolution profile
        sns.lineplot(data=score_all, x='resolution', y='score').set_title('Optimal cluster resolution profile')
        plt.show()

    if inplace:
        adata.obs[cluster_key] = clustering
        return res_max, score_max, score_all
    else:
        return res_max, score_max, score_all, clustering
      
