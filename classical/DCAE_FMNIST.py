from sklearn.datasets import load_digits
import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

import tensorflow as tf
import cv2

def DCAE(idx, lr, wt):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    idx_list = list(range(10))
    idx_list.remove(idx)
    vector_train = []
    vector_test = []
    vector_nontarget = []
    digit_target = x_train[np.where(y_train == idx)]
    for i in range(100):
        vector = cv2.resize(digit_target[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
        vector_train.append(vector)
    for i in range(100,200):
        vector = cv2.resize(digit_target[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
        vector_test.append(vector)
    for idx_nontarget in idx_list:
        digit_nontarget = x_train[np.where(y_train == idx_nontarget)]
        for i in range(100):
            vector = cv2.resize(digit_nontarget[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
            vector_nontarget.append(vector)
    vector_train = np.array(vector_train)
    vector_test = np.array(vector_test)
    vector_nontarget = np.array(vector_nontarget)

    vector_train = vector_train.reshape(100,1,16,16)
    vector_train = torch.from_numpy(vector_train)
    vector_test = vector_test.reshape(vector_test.shape[0],1,16,16)
    vector_test = torch.from_numpy(vector_test)
    vector_nontarget = vector_nontarget.reshape(vector_nontarget.shape[0],1,16,16)
    vector_nontarget = torch.from_numpy(vector_nontarget)

    train_dataloader = DataLoader(vector_train, batch_size=10, shuffle=True)

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()

            self.pool = nn.MaxPool2d(2, 2)

            # Encoder
            self.conv1 = nn.Conv2d(1, 2, kernel_size=3, bias=False, padding=1)
            self.bn1 = nn.BatchNorm2d(2, eps=1e-04, affine=False)
            self.conv2 = nn.Conv2d(2, 2, 3, bias=False, padding=1)
            self.bn2 = nn.BatchNorm2d(2, eps=1e-04, affine=False)
            self.fc1 = nn.Linear(2 * 4 * 4, 4, bias=False)

            # Decoder
            self.deconv1 = nn.ConvTranspose2d(1, 2, 3, bias=False, padding=1)
            self.bn3 = nn.BatchNorm2d(2, eps=1e-04, affine=False)
            self.deconv2 = nn.ConvTranspose2d(2, 2, 3, bias=False, padding=1)
            self.bn4 = nn.BatchNorm2d(2, eps=1e-04, affine=False)
            self.deconv3 = nn.ConvTranspose2d(2, 1, 3, bias=False, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.pool(F.leaky_relu(self.bn1(x)))
            x = self.conv2(x)
            x = self.pool(F.leaky_relu(self.bn2(x)))
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = x.view(x.size(0), 1, 2, 2)
            x = F.interpolate(F.leaky_relu(x), scale_factor=2)
            x = self.deconv1(x)
            x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
            x = self.deconv2(x)
            x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
            x = self.deconv3(x)
            x = torch.sigmoid(x)

            return x

    def train(epoch):
        for data in train_dataloader:
          autoencoder.train()
          tr_loss = 0
          # getting the training set
          train_vec = Variable(data)

          # clearing the Gradients of the model parameters
          optimizer.zero_grad()

          # prediction for training and validation set
          output_train = autoencoder(train_vec.float())

          # computing the training and validation loss
          scores = torch.sum((output_train - train_vec) ** 2, dim=tuple(range(1, output_train.dim())))
          loss_train = torch.mean(scores)

          # computing the updated weights of all the model parameters
          loss_train.backward()
          optimizer.step()
          tr_loss = loss_train.item()

    autoencoder = Autoencoder()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=wt)

    n_epochs = 250
    for epoch in range(n_epochs):
        train(epoch)

    test_vec = Variable(vector_test)
    output = autoencoder(test_vec.float())
    scores_test = torch.sum((output - test_vec) ** 2, dim=tuple(range(1, output.dim())))
    scores_test = scores_test.detach().numpy()

    nontarget_vec = Variable(vector_nontarget)
    output = autoencoder(nontarget_vec.float())
    scores_nontarget = torch.sum((output - nontarget_vec) ** 2, dim=tuple(range(1, output.dim())))
    scores_nontarget = scores_nontarget.detach().numpy()

    y_true = np.array([0]*len(scores_test)+[1]*len(scores_nontarget))
    y_score = np.append(scores_test,scores_nontarget)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_measure = auc(fpr,tpr)

    return auc_measure
