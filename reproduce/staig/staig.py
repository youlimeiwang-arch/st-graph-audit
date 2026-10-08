import argparse
import os
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.spatial.distance import cdist
from torch_geometric.nn import GCNConv
import numpy as np
from .utils import clustering
from sklearn import metrics
import scanpy as sc
from sklearn.cluster import KMeans
from .net import Encoder, MVmodel, SVmodel, drop_feature, dropout_adj, random_dropout_adj, Discriminator,multiple_dropout_average, training_graph_views, training_feature_views
from scipy.special import softmax
import pickle
import datetime
import matplotlib.pyplot as plt
from torch.autograd import Function
import umap
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from anndata import AnnData
import pickle
from sklearn.metrics import silhouette_score, davies_bouldin_score,calinski_harabasz_score
import tqdm
from .metrics import BatchKL
import harmonypy as hm
from torch_geometric.utils import to_torch_coo_tensor



def generate_pseudo_labels(img_emb, n_clusters=300):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_emb)
    pseudo_labels = kmeans.labels_
    return torch.tensor(pseudo_labels)


def adj_to_edge_index(adj):
    row, col = torch.where(adj != 0)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def convert_edge_probabilities(adj_matrix, edge_prob_matrix):
    row, col = torch.where(adj_matrix != 0)
    edge_probs = edge_prob_matrix[row, col]
    return edge_probs


def save_spatial_pair(adata, outfile, img_key="hires", size=1.6):
    """Stack ground_truth / domain vertically so legends do not collide."""
    colors = [c for c in ("ground_truth", "domain") if c in adata.obs]
    if not colors:
        raise ValueError("adata.obs has neither ground_truth nor domain")
    n = len(colors)
    max_cats = max(int(adata.obs[c].nunique()) for c in colors)
    legend_w = 2.6 if max_cats <= 12 else (3.8 if max_cats <= 24 else 5.4)
    fs = 9 if max_cats <= 10 else (8 if max_cats <= 24 else 7)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(6.8 + legend_w, 6.4 * n),
        squeeze=False,
        gridspec_kw={"hspace": 0.38},
    )
    has_img = bool(adata.uns.get("spatial")) and img_key
    for ax, color in zip(axes[:, 0], colors):
        adata.obs[color] = adata.obs[color].astype("category")
        if has_img:
            sc.pl.spatial(
                adata,
                img_key=img_key,
                size=size if size != 100 else 1.6,
                color=color,
                ax=ax,
                show=False,
                title=str(color),
                legend_loc="right margin",
                legend_fontsize=fs,
                frameon=False,
            )
        else:
            sc.pl.embedding(
                adata,
                basis="spatial",
                color=color,
                size=size if size == 100 else 80,
                ax=ax,
                show=False,
                title=str(color),
                legend_loc="right margin",
                legend_fontsize=fs,
                frameon=False,
            )
        n_cat = int(adata.obs[color].nunique())
        ncol = 1 if n_cat <= 14 else (2 if n_cat <= 36 else 3)
        leg = ax.get_legend()
        if leg is not None:
            handles = getattr(leg, "legend_handles", None) or leg.legendHandles
            labels = [t.get_text() for t in leg.texts]
            leg.remove()
            ax.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.04, 0.5),
                frameon=False,
                fontsize=fs,
                ncol=ncol,
                columnspacing=1.65,
                handletextpad=0.55,
                borderaxespad=0.2,
                markerscale=1.1,
            )
    os.makedirs(os.path.dirname(os.path.abspath(outfile)) or ".", exist_ok=True)
    fig.savefig(outfile, dpi=170, bbox_inches="tight", pad_inches=0.4, facecolor="white")
    plt.close(fig)
    print("wrote", outfile)



class STAIG:

    def __init__(self,args,config, single: bool = False, refine: bool=True):
        self.args = args
        self.single = single
        self.config = config
        self.learning_rate = config['learning_rate']
        self.num_hidden = config['num_hidden']
        self.num_proj_hidden = config['num_proj_hidden']
        self.activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
        self.base_model = ({'GCNConv': GCNConv})[config['base_model']]
        self.num_layers = config['num_layers']
        self.drop_feature_rate_1 = config['drop_feature_rate_1']
        self.drop_feature_rate_2 = config['drop_feature_rate_2']
        self.tau = config['tau']
        self.num_epochs = config['num_epochs']
        self.weight_decay = config['weight_decay']
        self.num_clusters = config['num_clusters']
        self.num_gene = config['num_gene']
        self.refine =refine
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.radius = 15
        self.tool = 'mclust' # mclust, leiden, and louvain
        self.bar_format = '{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        self.encoder = Encoder(self.num_gene, self.num_hidden, self.activation, base_model=self.base_model, k=self.num_layers).to(self.device)
        self.adata = None
        self.mask_slices = True

        if single:
            self.model = SVmodel(self.encoder, self.num_hidden, self.num_proj_hidden, self.tau).to(self.device).double()
            # self.model = SVmodel(self.encoder, self.num_hidden, self.num_proj_hidden, self.tau).to(self.device)
        else:
            self.model = MVmodel(self.encoder, self.num_hidden, self.num_proj_hidden, self.tau).to(self.device).double()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def train(self):
        print('=== prepare for training ===')
        if self.adata is None:
            raise ValueError("adata not load!")
        if self.single is False:
            features_matrix = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device).double()
        
            graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy()).to(self.device).double()
            edge_probabilities = torch.FloatTensor(self.adata.obsm['edge_probabilities'].copy()).to(self.device).double()
            if 'pseudo_labels' in self.adata.obs:
                pseudo_labels = torch.tensor(self.adata.obs['pseudo_labels'].cat.codes).to(self.device)
            else:                
                if 'k' in self.config:
                    pseudo_labels = generate_pseudo_labels(self.adata.obsm['img_emb'], self.config['k'])
                else:
                    pseudo_labels = generate_pseudo_labels(self.adata.obsm['img_emb'])
                pseudo_labels = pseudo_labels.to(self.device)

            edge_index = adj_to_edge_index(graph_neigh)
            edge_probs = convert_edge_probabilities(graph_neigh, edge_probabilities)

            start = datetime.datetime.now()
            prev = start
            print('=== train ===')
            for epoch in tqdm.tqdm(range(1, self.num_epochs + 1), bar_format=self.bar_format):
                self.model.train()
                self.optimizer.zero_grad()
                edge_index_1, edge_index_2 = training_graph_views(
                    edge_index, edge_probs, self.config, force_undirected=True
                )
                x_1, x_2 = training_feature_views(
                    features_matrix, self.drop_feature_rate_1, self.drop_feature_rate_2, self.config
                )
                z1 = self.model(x_1, edge_index_1)
                z2 = self.model(x_2, edge_index_2)
                loss = self.model.contrastive_loss_bias(z1, z2, graph_neigh, pseudo_labels)
                loss.backward()
                self.optimizer.step()
            torch.save(self.model.state_dict(), 'model.pt')

        if self.single:
            features_matrix = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device).double()
            edge_probabilities = torch.FloatTensor(self.adata.obsm['edge_probabilities'].copy()).to(self.device).double()
            graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy()).to(self.device).double()
            # features_matrix = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
            # edge_probabilities = torch.FloatTensor(self.adata.obsm['edge_probabilities'].copy()).to(self.device)
            # graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy()).to(self.device)

            if ('mask_neigh' in self.adata.obsm) and (self.mask_slices):
                print('Consider intra slice')
                mask_neigh = torch.FloatTensor(self.adata.obsm['mask_neigh'].copy()).to(self.device)
            else: 
                 mask_neigh = None
            edge_index = adj_to_edge_index(graph_neigh)
            edge_probs = convert_edge_probabilities(graph_neigh, edge_probabilities)

            start = datetime.datetime.now()
            prev = start
            print('=== train ===')
            for epoch in tqdm.tqdm(range(1, self.num_epochs + 1), bar_format=self.bar_format):

                self.model.train()
                self.optimizer.zero_grad()
                edge_index_1, edge_index_2 = training_graph_views(
                    edge_index, edge_probs, self.config, force_undirected=True
                )
                x_1, x_2 = training_feature_views(
                    features_matrix, self.drop_feature_rate_1, self.drop_feature_rate_2, self.config
                )
                z1 = self.model(x_1, edge_index_1)
                z2 = self.model(x_2, edge_index_2)
                loss = self.model.contrastive_loss(z1, z2, graph_neigh, mask=mask_neigh)
                loss.backward()
                self.optimizer.step()
            torch.save(self.model.state_dict(), 'model.pt')
    
    

    def eva(self):
        print("=== load ===")
        self.model.load_state_dict(torch.load('model.pt'))
        self.model.eval()
        if self.single:
            features_matrix = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device).double()
            graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy()).to(self.device).double()
        else:
            features_matrix = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device).double()
            graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy()).to(self.device).double()
        edge_index = adj_to_edge_index(graph_neigh)
        self.adata.obsm['emb'] = self.model(features_matrix, edge_index).detach().cpu().numpy()
        print(self.adata.obsm['emb'])
        print('embedding generated, go clustering')
    
    


    def cluster(self,label=True):
        if self.tool == 'mclust':
            clustering(self.adata, self.num_clusters, radius=self.radius, method=self.tool,
                       refinement=self.refine)  # For DLPFC dataset, we use optional refinement step.
        elif self.tool in ['leiden', 'louvain']:
            clustering(self.adata, self.num_clusters, radius=self.radius, method=self.tool, start=0.01, end=0.27, increment=0.005,
                       refinement=True)

        if label:
            print('calculate metric ARI')
            # calculate metric ARI
            ARI = metrics.adjusted_rand_score(self.adata.obs['domain'], self.adata.obs['ground_truth'])
            self.adata.uns['ari'] = ARI
            NMI = metrics.normalized_mutual_info_score(self.adata.obs['domain'], self.adata.obs['ground_truth'])
            self.adata.uns['nmi'] = NMI
            print('ARI:', ARI)
            print('NMI:', NMI)
        else:
            print("calculate SC and DB")
            SC = silhouette_score(self.adata.obsm['emb'], self.adata.obs['domain'])
            DB = davies_bouldin_score(self.adata.obsm['emb'], self.adata.obs['domain'])
            self.adata.uns['sc'] = SC
            self.adata.uns['db'] = DB
            print('SC:', SC)
            print('DB:', DB)
        if 'batch' in (self.adata.obs):
            BatchKL(self.adata)
            ILISI = hm.compute_lisi(self.adata.obsm['emb'], self.adata.obs[['batch']], label_colnames=['batch'])[:, 0]
            median_ILISI = np.median(ILISI)
            print(f'Median ILISI: {median_ILISI:.2f}')





    def draw_spatial(self,p=''):
        figdir = str(sc.settings.figdir or "figures")
        os.makedirs(figdir, exist_ok=True)
        out = os.path.join(figdir, f"show{p}{self.args.slide}.png")
        save_spatial_pair(self.adata, out)

    def draw_single_spatial(self):
        figdir = str(sc.settings.figdir or "figures")
        os.makedirs(figdir, exist_ok=True)
        out = os.path.join(figdir, f"show{self.args.slide}.png")
        save_spatial_pair(self.adata, out, img_key=None, size=100)

    def draw_umap(self):
        print('start umap')
        sc.pp.neighbors(self.adata, use_rep='emb')
        sc.tl.umap(self.adata)
        sc.pl.umap(self.adata, color='domain', show=True, save=str(self.args.slide)+ 'domain.png')
        sc.pl.umap(self.adata, color='batch', show=True, save=str(self.args.slide)+ '_batch.png')
        if self.args.label==True:
            sc.pl.umap(self.adata, color='ground_truth', show=True, save=str(self.args.slide)+ '_label.png')
        
        
    def draw_horizontal(self):
        adata_batch_0 = self.adata[self.adata.obs['batch'] == '0', :]
        sc.pl.embedding(adata_batch_0, basis="spatial", color="domain",
                        save='0.png')

        adata_batch_1 = self.adata[self.adata.obs['batch'] == '1', :]
        sc.pl.embedding(adata_batch_1, basis="spatial", color="domain",
                        save='1.png')


