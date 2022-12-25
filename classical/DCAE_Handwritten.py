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

def DCAE(idx, lr, wt):
    digits = load_digits()
    neg_list = list(range(10))
    neg_list.remove(idx)
    vector_train = []
    vector_test_pos = []
    vector_test_neg = []
    digit_pos = digits.data[np.where(digits.target == idx)]/16.0
    for i in range(100):
        vector_train.append(digit_pos[i].reshape((8,8)))
    for i in range(100,170):
        vector_test_pos.append(digit_pos[i].reshape((8,8)))
    for idx_neg in neg_list:
        digit_neg = digits.data[np.where(digits.target == idx_neg)]/16.0
        for i in range(70):
            vector_test_neg.append(digit_neg[i].reshape((8,8)))
    vector_train = np.array(vector_train)
    vector_test_pos = np.array(vector_test_pos)
    vector_test_neg = np.array(vector_test_neg)

    vector_train = vector_train.reshape(100,1,8,8)
    vector_train = torch.from_numpy(vector_train)
    vector_test_pos = vector_test_pos.reshape(vector_test_pos.shape[0],1,8,8)
    vector_test_pos = torch.from_numpy(vector_test_pos)
    vector_test_neg = vector_test_neg.reshape(vector_test_neg.shape[0],1,8,8)
    vector_test_neg = torch.from_numpy(vector_test_neg)

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
            self.fc1 = nn.Linear(2 * 2 * 2, 4, bias=False)

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
            x = F.interpolate(F.leaky_relu(x), scale_factor=1)
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

    test_pos_vec = Variable(vector_test_pos)
    output = autoencoder(test_pos_vec.float())
    scores_pos = torch.sum((output - test_pos_vec) ** 2, dim=tuple(range(1, output.dim())))
    scores_pos = scores_pos.detach().numpy()

    test_neg_vec = Variable(vector_test_neg)
    output = autoencoder(test_neg_vec.float())
    scores_neg = torch.sum((output - test_neg_vec) ** 2, dim=tuple(range(1, output.dim())))
    scores_neg = scores_neg.detach().numpy()

    y_true = np.array([0]*len(scores_pos)+[1]*len(scores_neg))
    y_score = np.append(scores_pos,scores_neg)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_measure = auc(fpr,tpr)

    return auc_measure
