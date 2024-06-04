import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score

 
def embedding(
        adata, 
        color='celltype', 
        color_map=None, 
        groupby='batch', 
        groups=None, 
        cond2=None, 
        v2=None, 
        save=None, 
        legend_loc='right margin', 
        legend_fontsize=None, 
        legend_fontweight='bold', 
        sep='_', 
        basis='X_umap',
        size=10,
        show=True,
    ):
    """
    plot separated embeddings with others as background
    
    Parameters
    ----------
    adata
        AnnData
    color
        meta information to be shown
    color_map
        specific color map
    groupby
        condition which is based-on to separate
    groups
        specific groups to be shown
    cond2
        another targeted condition
    v2
        another targeted values of another condition
    basis
        embeddings used to visualize, default is X_umap for UMAP
    size
        dot size on the embedding
    """
    
    if groups is None:
        groups = adata.obs[groupby].cat.categories
    for b in groups:
        adata.obs['tmp'] = adata.obs[color].astype(str)
        adata.obs['tmp'][adata.obs[groupby]!=b] = ''
        if cond2 is not None:
            adata.obs['tmp'][adata.obs[cond2]!=v2] = ''
            groups = list(adata[(adata.obs[groupby]==b) & 
                                (adata.obs[cond2]==v2)].obs[color].astype('category').cat.categories.values)
            size = min(size, 120000/len(adata[(adata.obs[groupby]==b) & (adata.obs[cond2]==v2)]))
        else:
            groups = list(adata[adata.obs[groupby]==b].obs[color].astype('category').cat.categories.values)
            size = min(size, 120000/len(adata[adata.obs[groupby]==b]))
        adata.obs['tmp'] = adata.obs['tmp'].astype('category')
        if color_map is not None:
            palette = [color_map[i] if i in color_map else 'gray' for i in adata.obs['tmp'].cat.categories]
        else:
            palette = None

        title = b if cond2 is None else v2+sep+b
        if save is not None:
            save_ = '_'+b+save
            show = False
        else:
            save_ = None
            show = True
        sc.pl.embedding(adata, color='tmp', basis=basis, groups=groups, title=title, palette=palette, size=size, save=save_,
                   legend_loc=legend_loc, legend_fontsize=legend_fontsize, legend_fontweight=legend_fontweight, show=show)
        del adata.obs['tmp']
        del adata.uns['tmp_colors']
        
        

def plot_confusion(y, y_pred, save=None, cmap='Blues'):
    """
    Plot confusion matrix
    
    Parameters
    ----------
    y
        ground truth labels
    y_pred 
        predicted labels
    save
        save the figure
    cmap
        color map
        
    Return
    ------
    F1 score
    NMI score
    ARI score
    """
    
    y_class, pred_class_ = np.unique(y), np.unique(y_pred)

    cm = confusion_matrix(y, y_pred)
    f1 = f1_score(y, y_pred, average='micro')
    nmi = normalized_mutual_info_score(y, y_pred)
    ari = adjusted_rand_score(y, y_pred)
    
    cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]

    plt.figure(figsize=(14, 14))
    sns.heatmap(cm, xticklabels=y_class, yticklabels=pred_class,
                    cmap=cmap, square=True, cbar=False, vmin=0, vmax=1)

    plt.xticks(rotation=45, horizontalalignment='right') #, fontsize=14)
    plt.yticks(fontsize=14, rotation=0)
    plt.ylabel('Leiden cluster', fontsize=18)
    
    if save:
        plt.save(save, bbox_inches='tight')
    else:
        plt.show()
        
