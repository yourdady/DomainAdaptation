''' 
@project DomainAdaptation
@author Peng
@file visulization.py.py
@time 2018-07-23
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
pca = PCA(n_components=2)
tsne = TSNE(n_components=2)
def plot_2d(X_src, y_src, X_tar, y_tar, title=None, n_classes = 10):
    # X_src = tsne.fit_transform(X_src)
    # X_tar = tsne.fit_transform(X_tar)
    # X_src = pca.fit_transform(X_src)
    # X_tar = pca.fit_transform(X_tar)
    fig = plt.figure(figsize=(8, 8))
    plt.axis([0, 2, 0, 1])
    ax = fig.add_subplot(1, 1, 1)
    x_min, x_max = np.min(X_src, axis=0), np.max(X_src, axis=0)
    X = (X_src - x_min) / (x_max - x_min)

    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1],
                # str(y_src[i]),
                str('*'),
                color=plt.cm.Set1(y_src[i] / n_classes),
                fontdict={'weight': 'bold', 'size': 9})

    x_min, x_max = np.min(X_tar, axis=0), np.max(X_tar, axis=0)
    X = (X_tar - x_min) / (x_max - x_min)
    for i in range(X.shape[0]):
        ax.text(X[i, 0]+1, X[i, 1],
                # str(y_tar[i]),
                str('*'),
                color=plt.cm.Set1(y_tar[i] / n_classes),
                fontdict={'weight': 'bold', 'size': 9})
    plt.show()