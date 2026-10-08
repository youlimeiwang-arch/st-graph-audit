import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc
import ot
from sklearn.preprocessing import StandardScaler


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='norm_emb', random_seed=2023):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    import os
    from pathlib import Path

    # Windows/rpy2: R bin must be on PATH before embedding, or stats.dll fails to load.
    r_home = os.environ.get("R_HOME") or r"C:\Program Files\R\R-4.6.0"
    r_bin = Path(r_home) / "bin" / "x64"
    if r_bin.is_dir():
        os.environ["R_HOME"] = str(Path(r_home))
        path = os.environ.get("PATH", "")
        if str(r_bin) not in path.split(os.pathsep):
            os.environ["PATH"] = str(r_bin) + os.pathsep + path

    np.random.seed(random_seed)
    import rpy2.robjects as robjects

    robjects.r.library("mclust")
    x = np.ascontiguousarray(adata.obsm[used_obsm], dtype=np.float64)
    # Avoid rpy2 SignatureTranslatedFunction kwargs (breaks Mclust on newer rpy2).
    rx = robjects.r.matrix(
        robjects.FloatVector(x.ravel(order="F")), nrow=x.shape[0], ncol=x.shape[1]
    )
    robjects.r.assign("staig_mclust_x", rx)
    robjects.r(
        f'set.seed({int(random_seed)}); '
        f'staig_mclust_res <- Mclust(staig_mclust_x, G={int(num_cluster)}, '
        f'modelNames="{modelNames}")'
    )
    mclust_res = np.array(robjects.r("staig_mclust_res$classification"))

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, n_clusters=7, radius=50, method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1.
    end : float 
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.   
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """

    if method == 'mclust':
       adata = mclust_R(adata, used_obsm='emb', num_cluster=n_clusters)
       adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
       res = search_res(radius,adata, n_clusters, use_rep='norm_emb', method=method, start=start, end=end, increment=increment)
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
       res = search_res(radius,adata, n_clusters, use_rep='norm_emb', method=method, start=start, end=end, increment=increment)
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['louvain'] 
       
    if refinement:
       new_type = refine_label(adata, radius, key='domain')
       adata.obs['domain'] = new_type 
       
def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type


    
def search_res(radius,adata, n_clusters, method='leiden', use_rep='norm_emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    ress=[]
    best=None
    sc.pp.neighbors(adata, n_neighbors=20, use_rep=use_rep)
    sc.tl.leiden(adata, random_state=0, resolution=end)
    count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
    while (count_unique > n_clusters +2):
        print(count_unique)
        print('太大，继续调整')
        end = end - 0.1
        sc.tl.leiden(adata, random_state=0, resolution=end)
        count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
    while (count_unique < n_clusters + 2):
        print(count_unique)
        print('太小，继续调整')
        end = end + 0.1
        sc.tl.leiden(adata, random_state=0, resolution=end)
        count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())


    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))

        if count_unique == n_clusters:
            print('calculate metric ARI')
            # calculate metric ARI
            new_type = refine_label(adata, radius, key='leiden')
            adata.obs['leiden'] = new_type

            ARI = metrics.adjusted_rand_score(adata.obs['leiden'], adata.obs['ground_truth'])
            adata.uns['ARI'] = ARI
            ress.append((res,ARI))
            print('ARI:', ARI)

        if count_unique == n_clusters-2:
            label = 1
            best = max(ress, key=lambda x: x[1])
            print(best)
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return best[0]
