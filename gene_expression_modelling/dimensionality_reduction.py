import pandas as pd, sklearn.decomposition as dcp, sklearn.manifold as manifold

def pca(df, n_components=None, kernel='linear'):
    pca = dcp.KernelPCA(n_components=n_components, kernel=kernel)
    in_pca_basis = pd.DataFrame(pca.fit_transform(df.values), index=df.index)

    return pca, in_pca_basis

def tsne(df, n_components=None, pca_dim=50, pca_kernel='linear'):
    p = None
    if pca_dim > 0 and df.shape[1] > pca_dim:
        p, df = pca(df, n_components=pca_dim, kernel=pca_kernel)

    tsne = manifold.TSNE(n_components=n_components)
    in_emb_space = pd.DataFrame(tsne.fit_transform(df.values), index=df.index)

    return tsne, in_emb_space, p
