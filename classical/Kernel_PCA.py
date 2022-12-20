import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import load_digits
import tensorflow as tf
import cv2

def Kernel_PCA(dataset,idx):
    '''
    Kernel PCA returning AUC measure of the test dataset
    --------
    Args :
        dataset : Dataset for one-class classification "Handwritten" or "FMNIST"
        idx : Index of the dataset to be trained/tested for one-class classification
    --------
    Return :
        auc_measure : AUC measure of the test dataset
    '''

    vector_target_train = []
    vector_target_test = []
    vector_nontarget = []
    idx_list = list(range(10))
    idx_list.remove(idx)

    if dataset == "Handwritten":
        digits = load_digits()
        target_digit = digits.data[np.where(digits.target == idx)]

        for i in range(100):
            vector_target_train.append(np.array(target_digit[i]))
        for i in range(100,170):
            vector_target_test.append(np.array(target_digit[i]))
        for idx_nontarget in idx_list:
            nontarget_digit = digits.data[np.where(digits.target == idx_nontarget)]
            for i in range(70):
                vector_nontarget.append(np.array(nontarget_digit[i]))

        vector_target_train= np.array(vector_target_train)
        vector_target_test= np.array(vector_target_test)
        vector_nontarget= np.array(vector_nontarget)

        perm_target = np.random.permutation(70)
        perm_nontarget = np.random.permutation(630)
        vector_target_val = vector_target_test[perm_target[:7]]
        vector_nontarget_val = vector_nontarget[perm_nontarget[:63]]

    elif dataset == "FMNIST":
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        target_digit = x_train[np.where(y_train == idx)]

        for i in range(100):
            vector = cv2.resize(target_digit[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
            vector_target_train.append(vector)
        for i in range(100,200):
            vector = cv2.resize(target_digit[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
            vector_target_test.append(vector)
        for idx_nontarget in idx_list:
            nontarget_digit = x_train[np.where(y_train == idx_nontarget)]
            for i in range(100):
                vector = cv2.resize(nontarget_digit[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
                vector_nontarget.append(vector)

        vector_target_train= np.array(vector_target_train)
        vector_target_test= np.array(vector_target_test)
        vector_nontarget= np.array(vector_nontarget)
        perm_target = np.random.permutation(100)
        perm_nontarget = np.random.permutation(900)
        vector_target_val = vector_target_test[perm_target[:10]]
        vector_nontarget_val = vector_nontarget[perm_nontarget[:90]]


    else:
        raise ValueError(
            "Handwritten digit and Fashion MNIST datasets are supported"
        )

    auc_val = 0.0
    gamma_opt = 0
    for gamma in np.logspace(-10, -1, num=10, base=2):
        kernel_pca = KernelPCA(n_components=None, kernel="rbf", gamma=gamma, fit_inverse_transform=True, alpha=0.1)
        kernel_pca.fit(vector_target_train)
        vector_target_train_PCA = kernel_pca.inverse_transform(kernel_pca.transform(vector_target_train))
        vector_target_test_PCA = kernel_pca.inverse_transform(kernel_pca.transform(vector_target_val))
        vector_nontarget_PCA = kernel_pca.inverse_transform(kernel_pca.transform(vector_nontarget_val))
        vector_target_mse = np.square(np.subtract(vector_target_val,vector_target_test_PCA)).mean(axis=1)
        vector_nontarget_mse = np.square(np.subtract(vector_nontarget_val,vector_nontarget_PCA)).mean(axis=1)

        y_true = np.array([0]*len(vector_target_mse)+[1]*len(vector_nontarget_mse))
        y_score = np.r_[vector_target_mse,vector_nontarget_mse]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_new = auc(fpr,tpr)
        if auc_new > auc_val :
            auc_val = auc_new
            gamma_opt = gamma

    kernel_pca = KernelPCA(n_components=None, kernel="rbf", gamma=gamma_opt, fit_inverse_transform=True, alpha=0.1)
    kernel_pca.fit(vector_target_train)
    vector_target_train_PCA = kernel_pca.inverse_transform(kernel_pca.transform(vector_target_train))
    vector_target_test_PCA = kernel_pca.inverse_transform(kernel_pca.transform(vector_target_test))
    vector_nontarget_PCA = kernel_pca.inverse_transform(kernel_pca.transform(vector_nontarget))
    vector_target_mse = np.square(np.subtract(vector_target_test,vector_target_test_PCA)).mean(axis=1)
    vector_nontarget_mse = np.square(np.subtract(vector_nontarget,vector_nontarget_PCA)).mean(axis=1)

    y_true = np.array([0]*len(vector_target_mse)+[1]*len(vector_nontarget_mse))
    y_score = np.r_[vector_target_mse,vector_nontarget_mse]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_measure = auc(fpr,tpr)

    return auc_measure


if __name__ == '__main__':
    d = "FMNIST"
    idx = 0
    a = Kernel_PCA(d,idx)
    print(a)
