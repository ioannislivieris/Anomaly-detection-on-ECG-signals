# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Basic libraries
#
import os
import numpy as np



# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Sklearn library
#
from sklearn.manifold      import TSNE
from sklearn.decomposition import TruncatedSVD

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Umap library
#
import umap


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Visualization libraries
#
import matplotlib.pyplot as plt



def plot_latent_space(z_run, labels, figsize = (15, 4), folder_name='./images'):
    """
    Given latent variables for all timeseries, and output of k-means, 
    run PCA and tSNE on latent vectors and color the points using cluster_labels.
    :param z_run: Latent vectors for all input tensors
    :param labels: Cluster labels for all input tensors
    :param engine: plotly/matplotlib
    :param download: If true, it will download plots in `folder_name`
    :param folder_name: Download folder to dump plots
    :return:
    """

    if (z_run.shape[1] > 2):
        z_run_pca  = TruncatedSVD(n_components=2).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)
        z_run_umap = umap.UMAP(n_neighbors = 30).fit_transform( z_run )
        
        fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = figsize)
        
        idx = np.where(labels == 'NORMAL')
        ax[0].scatter(z_run_pca[idx, 0], z_run_pca[idx, 1], marker='o', color='tab:blue',   s=10)
        idx = np.where(labels == 'WARNING')
        ax[0].scatter(z_run_pca[idx, 0], z_run_pca[idx, 1], marker='s', color='tab:orange', s=15)
        idx = np.where(labels == 'CRITICAL')
        ax[0].scatter(z_run_pca[idx, 0], z_run_pca[idx, 1], marker='x', color='tab:red',    s=20)
        #
        ax[0].set_title('PCA on latent-space')
        ax[0].legend(['NORMAL', 'WARNING', 'CRITICAL'], frameon = False, fontsize = 14)


        
        idx = np.where(labels == 'NORMAL')
        ax[1].scatter(z_run_tsne[idx, 0], z_run_tsne[idx, 1], marker='o', color='tab:blue',   s=10)
        idx = np.where(labels == 'WARNING')
        ax[1].scatter(z_run_tsne[idx, 0], z_run_tsne[idx, 1], marker='s', color='tab:orange', s=15)
        idx = np.where(labels == 'CRITICAL')
        ax[1].scatter(z_run_tsne[idx, 0], z_run_tsne[idx, 1], marker='x', color='tab:red',    s=20)
        #
        ax[1].set_title('SNE on latent-space')
        ax[1].legend(['NORMAL', 'WARNING', 'CRITICAL'], frameon = False, fontsize = 14)

        
        idx = np.where(labels == 'NORMAL')
        ax[2].scatter(z_run_umap[idx, 0], z_run_umap[idx, 1], marker='o', color='tab:blue',   s=10)
        idx = np.where(labels == 'WARNING')
        ax[2].scatter(z_run_umap[idx, 0], z_run_umap[idx, 1], marker='s', color='tab:orange', s=15)
        idx = np.where(labels == 'CRITICAL')
        ax[2].scatter(z_run_umap[idx, 0], z_run_umap[idx, 1], marker='x', color='tab:red',    s=20)
        ax[2].set_title('Umap on latent-space')
        ax[2].legend(['NORMAL', 'WARNING', 'CRITICAL'], frameon = False, fontsize = 14)
        
        plt.show()
    else:
        plt.figure( figsize = figsize)
        
        idx = np.where(labels == 'NORMAL')
        plt.scatter(z_run[idx, 0], z_run[idx, 1], marker='o', color='tab:blue',   s=10)
        idx = np.where(labels == 'WARNING')
        plt.scatter(z_run[idx, 0], z_run[idx, 1], marker='s', color='tab:orange', s=15)
        idx = np.where(labels == 'CRITICAL')
        plt.scatter(z_run[idx, 0], z_run[idx, 1], marker='x', color='tab:red',    s=20)
        plt.show()
