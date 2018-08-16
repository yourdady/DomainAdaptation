
''' 
@project DomainAdaptation
@author Peng
@file BDA.py
@time 2018-07-16
'''

import numpy as np
import scipy as scp
import scipy.sparse
import math
from scipy.sparse import linalg
from sklearn.metrics import accuracy_score
class BDA():

    """Balanced Domain Adaptation.
        Parameters
        ----------
        dim : int, number of new features.

        kernel_param : float, hyper param of kernel function.

        kernel_typpe : string, type of kernel function.

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights.

        mode : 'W-BDA' or None, set mode='W-BDA' for unbalanced data.

    """
    def __init__(self, dim ,kernel_param=1, kernel_type='rbf', mode = None):
        self.dim = dim
        self.kernelparam = kernel_param
        self.kerneltype = kernel_type
        self.mode = mode
    def fit_transform(self, X_src, Y_src, X_tar, Y_tar, X_tar_l = None, Y_tar_l = None, lam=0.1 ,X_o = None, mu=0.5, iterations = 10,
                      classifier = None):
        """fit_transform.
            Parameters
            ----------
            X_src : 2d array shape of [n_src_samples, n_features].

            Y_src : 1d array shape of [n_src_samples].
            
            X_tar : 2d array shape of [n_tar_samples, n_features].
            
            Y_tar : 1d array shape of [n_tar_samples].
            
            X_tar_l : 2d array shape of [n_tar_l_samples, n_features]. When limited labeled data 
            is available in target source, they could be used to improve the performance.
            
            Y_tar_l : 1d array shape of [n_tar_l_samples]. When limited labeled data 
            is available in target source, they could be used to improve the performance.
            
            lam : float, hyper parameter for BDA.
            
            mu : float, hyper parameter for BDA, when mu=0.5, BDA <=> JDA, when mu=1, BDA <=> TCA.
            
            iterations : int.
            
            classifier : classifier for adaptation.
    
        """
        n_tar_l = 0
        n_src = X_src.shape[0]
        n_tar = X_tar.shape[0]
        if X_tar_l is not None and Y_tar_l is not None:
            n_tar_l = X_tar_l.shape[0]

        X_src = self.zscore(X_src)
        if X_tar_l is not None and Y_tar_l is not None:
            X_tar_zscore = self.zscore(np.concatenate((X_tar,X_tar_l)))
            X_tar = X_tar_zscore[:n_tar]
            X_tar_l = X_tar_zscore[n_tar:]

        else:
            X_tar = self.zscore(X_tar)

        X_src[np.isnan(X_src)] = 0
        X_tar[np.isnan(X_tar)] = 0
        if X_tar_l is not None and Y_tar_l is not None:
            X_tar_l[np.isnan(X_tar_l)] = 0

        X = np.hstack((np.transpose(X_src), np.transpose(X_tar)))
        if X_tar_l is not None and Y_tar_l is not None:
            X = np.hstack((X, np.transpose(X_tar_l)))
        X = np.dot(X, np.diag(1.0 / np.sqrt(np.sum(X * X,axis=0))))
        m,n = X.shape
        K_o = None
        if X_o is not None:
            X_o = X_o.T
            X_o = np.dot(X_o, np.diag(1.0 / np.sqrt(np.sum(X_o * X_o, axis=0))))
            K_o = self.get_kernel(self.kerneltype, self.kernelparam, X, X_o)
        e = np.vstack((1./n_src * np.ones((n_src,1),dtype=np.float32),
                       -1./(n_tar+n_tar_l) * np.ones((n_tar+n_tar_l,1),dtype=np.float32)))
        C = np.max(Y_tar) + 1
        M = np.dot(e ,e.T ) * C
        Y_tar_pseudo = []
        Z = None
        Z_o = None

        for T in range(iterations):
            N = np.zeros((n, n))
            if len(Y_tar_pseudo) != 0:
                for cls in range(C):
                    e = np.zeros((n,1))
                    ps = 1
                    pt = 1
                    if self.mode == 'W-BDA':
                        ps = len(np.where(Y_src == cls))/len(Y_src)
                        pt = len(np.where(Y_tar_pseudo == cls))/len(Y_tar_pseudo)
                        if X_tar_l is not None and Y_tar_l is not None:
                            pt = (len(np.where(Y_tar_pseudo == cls)) + len(np.where(Y_tar_l == cls))) / \
                                 (len(Y_tar_pseudo) + n_tar_l)
                    index = np.where(Y_src == cls)
                    e[index] = math.sqrt(ps) / len(index[0])
                    if X_tar_l is not None and Y_tar_l is not None:
                        index = np.where(np.concatenate((np.array(Y_tar_pseudo), Y_tar_l)) == cls)
                    else:
                        index = np.where(Y_tar_pseudo == cls)
                    e[index[0] + n_src] = -math.sqrt(pt) / len(index[0])
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)
            M = mu*M + (1-mu)*N
            H = np.eye(n) - 1 / n * np.ones((n, n))
            M = M / np.sqrt(np.sum(np.diag(np.dot(M.T, M))))

            if self.kerneltype=='primal':
                A = np.dot(np.dot(X, M), np.transpose(X))
                B = np.dot(np.dot(X, H), np.transpose(X))
                A[np.isinf(A)] = 0
                B[np.isinf(B)] = 0

                val,A = scp.sparse.linalg.eigs(A + lam * np.eye(m),
                                              self.dim,
                                              B,
                                              which='SM')
                return np.dot(np.transpose(A), X)
            else:
                K = self.get_kernel(self.kerneltype, self.kernelparam, X)

                #scipy eigs AV=BVD特征值经常不收敛，直接计算pinv(B)*A的特征值

                val, A = np.linalg.eig(np.dot(np.linalg.pinv(np.dot(np.dot(K, H), K.T)),
                                              np.dot(np.dot(K, M), K.T) + lam * np.eye(n)))
                eig_values = val.reshape(len(val), 1)
                index_sorted = np.argsort(-eig_values, axis=0)
                A = A[:, index_sorted]
                A = A.reshape((A.shape[0], A.shape[1]))

                if X_o is not None:
                    jda_o = np.dot(np.transpose(A), K_o)
                    Z_o = np.dot(jda_o, np.diag(1.0/np.sqrt(np.sum(np.multiply(jda_o,jda_o), 0))))

                Z = np.dot(np.transpose(A), K)
                Z = np.dot(Z, np.diag(1.0/np.sqrt(np.sum(np.multiply(Z,Z), 0))))
                Z_src = Z.T[:n_src]
                Z_tar = Z.T[n_src:n_src + n_tar]
                classifier.fit(Z_src,Y_src)
                Y_tar_pseudo = classifier.predict(Z_tar)
                acc = accuracy_score(Y_tar, Y_tar_pseudo)
                # acc2 = accuracy_score(np.concatenate((Y_tar,Y_tar_l)),
                #                                      np.concatenate((np.array(Y_tar_pseudo),Y_tar_l)))
                print('{} iterations accuracy: {}'.format(T,acc))
                # print('{} iterations accuracy2: {}'.format(T,acc2))

        return Z, Z_o


    def get_kernel(self, kerneltype, kernelparam, x1, x2=None):

        dim, n1 = x1.shape
        K = None
        if x2 is not None:
            n2 = x2.shape[1]
        if kerneltype == 'linear':
            if x2 is not None:
                K = np.dot(x1.T, x2)
            else:
                K = np.dot(x1.T, x1)
        elif kerneltype == 'poly':
            if x2 is not None:
                K = np.power(np.dot(x1.T, x2), kernelparam)
            else:
                K = np.power(np.dot(x1.T, x1), kernelparam)
        elif kerneltype == 'rbf':
            if x2 is not None:
                sum_x2 = np.sum(np.multiply(x2.T, x2.T), axis=1)
                sum_x1 = np.sum(np.multiply(x1.T, x1.T), axis=1)
                sum_x1 = sum_x1.reshape((len(sum_x1), 1))


                L2= np.tile(sum_x1, (1, n2)) + np.tile(sum_x2.T, (n1, 1)) - 2 * np.dot(x1.T,x2)

                K = np.exp(-1 * (
                    L2) / (dim * 2 * kernelparam))
            else:
                P = np.sum(np.multiply(x1.T, x1.T), axis=1)
                P = P.reshape((len(P),1))
                K = np.exp(
                    -1 * (np.tile(P.T, (n1, 1)) + np.tile(P, (1, n1)) -
                          2 * np.dot(x1.T, x1)) / (dim * 2 * kernelparam))
        return K


    def zscore(self, X):
        tmp = X / np.tile(np.sum(X,1).reshape((len(X),1)),(1,X.shape[1]))
        return (tmp-np.mean(tmp,0))/np.std(tmp,0)