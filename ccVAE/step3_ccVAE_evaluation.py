import numpy as np
import scanpy as sc
import anndata as ad
import torch
import anndata
import matplotlib.pyplot as plt
#from typing import Union

#from .step2_ccVAE_utils import label_encoder
from .step3_ccVAE_metrics import graph_connectivity, entropy_batch_mixing, knn_purity, asw, ari, nmi
#from .step2_ccVAE_MainModel import ccVAE
#from .step2_run_ccVAE import CCVAE
#from .step2_ccVAE_Trainer import ccVAETrainer

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)
np.set_printoptions(precision=2, edgeitems=7)


class CCVAE_EVAL:
    def __init__(
            self,
            adata_latent: anndata.AnnData,
            adata_ccPairs: anndata.AnnData,
            adata_rest: anndata.AnnData,
            condition_key: str = None,
            cell_type_key: str = None,
            n_neighbors=8,
    ):
        self.adata_ccPairs = adata_ccPairs
        self.adata_rest = adata_rest
        self.cell_type_names = None
        self.batch_names = None
        # Concatenate the two AnnData objects
        adata = ad.concat([self.adata_ccPairs, self.adata_rest], join='outer')
        
        if cell_type_key is not None:
            self.cell_type_names = adata.obs[cell_type_key].tolist()
        if condition_key is not None:
            self.batch_names = adata.obs[condition_key].tolist()

        self.adata_latent = adata_latent
        self.adata_latent.obs_names = adata.obs_names
        if self.cell_type_names is not None:
            self.adata_latent.obs['celltype'] = self.cell_type_names
        if self.batch_names is not None:
            self.adata_latent.obs['batch'] = self.batch_names
        del adata
        
        sc.pp.neighbors(self.adata_latent, n_neighbors=n_neighbors)
        sc.tl.umap(self.adata_latent)

    def plot_latent(self,
                    show=True,
                    save=False,
                    dir_path=None,
                    ):
        if save:
            show=False
            if dir_path is None:
                save = False

        color = [
            'celltype' if self.cell_type_names is not None else None,
            'batch' if self.batch_names is not None else None,
        ]
        sc.pl.umap(self.adata_latent,
                   color=color,
                   frameon=False,
                   wspace=0.6,
                   show=show)
        if save:
            plt.savefig(f'{dir_path}_batch.png', bbox_inches='tight')


    def plot_embedding(self,
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
      adata = self.adata_latent
      if groups is None:
          groups = self.adata_latent.obs[groupby].cat.categories
      for b in groups:
          self.adata_latent.obs['tmp'] = self.adata_latent.obs[color].astype(str)
          self.adata_latent.obs['tmp'][self.adata_latent.obs[groupby]!=b] = ''
          if cond2 is not None:
              self.adata_latent.obs['tmp'][self.adata_latent.obs[cond2]!=v2] = ''
              groups = list(self.adata_latent[(self.adata_latent.obs[groupby]==b) & 
                                  (self.adata_latent.obs[cond2]==v2)].obs[color].astype('category').cat.categories.values)
              size = min(size, 120000/len(self.adata_latent[(self.adata_latent.obs[groupby]==b) & (self.adata_latent.obs[cond2]==v2)]))
          else:
              groups = list(self.adata_latent[self.adata_latent.obs[groupby]==b].obs[color].astype('category').cat.categories.values)
              size = min(size, 120000/len(self.adata_latent[self.adata_latent.obs[groupby]==b]))
          self.adata_latent.obs['tmp'] = self.adata_latent.obs['tmp'].astype('category')
          if color_map is not None:
              palette = [color_map[i] if i in color_map else 'gray' for i in self.adata_latent.obs['tmp'].cat.categories]
          else:
              palette = None
    
          title = b if cond2 is None else v2+sep+b
          if save is not None:
              save_ = '_'+b+save
              show = False
          else:
              save_ = None
              show = True
          sc.pl.embedding(self.adata_latent, color='tmp', basis=basis, groups=groups, title=title, palette=palette, size=size, save=save_,
                     legend_loc=legend_loc, legend_fontsize=legend_fontsize, legend_fontweight=legend_fontweight, show=show)
          del self.adata_latent.obs['tmp']
          del self.adata_latent.uns['tmp_colors']
        
        
    def get_gc(self, verbose=True):
        gc_score = graph_connectivity(
            adata=self.adata_latent,
            label_key='batch'
        )
        if verbose:
            print("graph_connectivity-Score: %0.2f" % gc_score)
        return gc_score

    def get_ebm(self, n_neighbors=50, n_pools=50, n_samples_per_pool=100, verbose=True):
        ebm_score = entropy_batch_mixing(
            adata=self.adata_latent,
            label_key='batch',
            n_neighbors=n_neighbors,
            n_pools=n_pools,
            n_samples_per_pool=n_samples_per_pool
        )
        if verbose:
            print("Entropy of Batchmixing-Score: %0.2f" % ebm_score)
        return ebm_score

    def get_knn_purity(self, n_neighbors=50, verbose=True):
        knn_score = knn_purity(
            adata=self.adata_latent,
            label_key='celltype',
            n_neighbors=n_neighbors
        )
        if verbose:
            print("KNN Purity-Score:  %0.2f" % knn_score)
        return knn_score

    def get_asw(self, verbose=True):
        asw_score_batch, asw_score_cell_types = asw(adata=self.adata_latent, label_key='celltype', batch_key='batch')
        if verbose:
          print("ASW on batch:", asw_score_batch)
          print("ASW on celltypes:", asw_score_cell_types)
        return asw_score_batch, asw_score_cell_types

    def get_ari(self, implementation=None, verbose=True):
        ari_score = ari(adata=self.adata_latent, cluster_key='batch', label_key='celltype', implementation=implementation)
        if verbose:
          print("ARI score:", ari_score)
        return ari_score

    def get_nmi(self, verbose=True):
        nmi_score = nmi(adata=self.adata_latent, label_key='celltype')
        if verbose:
          print("NMI score:", nmi_score)
        return nmi_score

      
    def get_Batch_score(self):
        gc = self.get_gc(verbose=False)
        ebm = self.get_ebm(verbose=False)
        asw_batch, _ = self.get_asw(verbose=False)
        score = gc + ebm + asw_batch
        print("Batch-correction Score graph_connectivity+EBM+ASW_batch, graph_connectivity, EBM, ASW_batch: %0.2f, %0.2f, %0.2f, %0.2f" % (score, gc, ebm, asw_batch))
        return score
      
      
    def get_Bio_score(self, implementation=None, n_neighbors=50):
        nmi = self.get_nmi(verbose=False)
        ari = self.get_ari(verbose=False, implementation=implementation)
        _, asw_cell_types = self.get_asw(verbose=False)
        knn = self.get_knn_purity(n_neighbors=n_neighbors,verbose=False)
        score = nmi + ari + asw_cell_types + knn
        print("Bio-conservation Score NMI+ARI+ASW_bio+KNN, NMI, ARI, asw_cell_types, KNN: %0.2f, %0.2f, %0.2f, %0.2f, %0.2f" % (score, nmi, ari, asw_cell_types, knn))
        return score
      
    def get_latent_score(self):
        Batch_score = get_Batch_score()
        Bio_score = get_Bio_score()
        score = Batch_score + Bio_score
        print("Latent-Space Score Batch_score+Bio_score, Batch_score, Bio_score: %0.2f, %0.2f, %0.2f" % (score, Batch_score, Bio_score))
        return score
      
      
  
