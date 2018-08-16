from code.BDA import BDA
import scipy.io
import numpy as np
from sklearn.svm import SVC
DATA_PATH = '../data/'
C_path = DATA_PATH + 'Caltech10_SURF_L10.mat'
W_path = DATA_PATH + 'webcam_SURF_L10.mat'
A_path = DATA_PATH + 'amazon_SURF_L10.mat'
D_path = DATA_PATH + 'dslr_SURF_L10.mat'

COIL1_PATH = DATA_PATH + 'COIL_1.mat'
COIL2_PATH = DATA_PATH + 'COIL_2.mat'

C = scipy.io.loadmat(C_path)
W = scipy.io.loadmat(W_path)
A = scipy.io.loadmat(A_path)
D = scipy.io.loadmat(D_path)

coil1 = scipy.io.loadmat(COIL1_PATH)
coil2 = scipy.io.loadmat(COIL2_PATH)



def read_coil1():
    X_src = np.swapaxes(np.array(coil1['X_src']),1,0)
    X_tar = np.swapaxes(np.array(coil1['X_tar']),1,0)
    Y_src = np.ravel(np.array(coil1['Y_src']))-1
    Y_tar = np.ravel(np.array(coil1['Y_tar']))-1
    index = np.argsort([i%36 for i in range(len(Y_tar))])
    n_src = len(X_src)
    n_tar = len(X_tar)
    X_src = X_src[index]
    X_tar = X_tar[index]
    Y_src = Y_src[index]
    Y_tar = Y_tar[index]

    return X_src, X_tar, Y_src, Y_tar, n_src, n_tar


def test_coil1():
    X_src, X_tar, Y_src, Y_tar, n_src, n_tar = read_coil1()

    bda = BDA(dim=200,kernel_param=1/200.0,kernel_type='rbf')
    clf = SVC(kernel='rbf', gamma=1/200.0)
    n_tar_l = int(X_tar.shape[0]/3)
    X_tar_l = X_tar[:n_tar_l]
    Y_tar_l = Y_tar[:n_tar_l]
    X_tar = X_tar[n_tar_l:]
    Y_tar = Y_tar[n_tar_l:]
    Z = bda.fit_transform(X_src,Y_src,X_tar,Y_tar,X_tar_l,Y_tar_l,classifier=clf,mu=0.1)


    # bda.fit_transform(X_src,Y_src,X_tar,Y_tar,classifier=clf,mu=0.1)

if __name__ == '__main__':
    test_coil1()
