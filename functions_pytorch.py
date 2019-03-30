#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:56:22 2018

@author: will
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

def plot_batch(X1,X2):
    n = X1.shape[0]
    fig, ax = plt.subplots(nrows=n, ncols=2,figsize=(7,25))
    fig.tight_layout()
    for i in range(n):
        ax[i][0].tick_params(axis='both',which='both',bottom=False,top=False,labelbottom=False)
        ax[i][0].imshow(X1[i]/2+0.5)
        ax[i][1].tick_params(axis='both',which='both',bottom=False,top=False,labelbottom=False)
        ax[i][1].imshow(X2[i]/2+0.5)
        
class TripletGenerator(Dataset):
    def __init__(self, Ids, newWhale, transFun, p=0.125):
        self.Ids = Ids # df with first column being w ID, second being list of imgs for each w
        self.newWhale = newWhale # a list of img for new whale
        self.n_newWhale = len(newWhale) - 1
        self.transFun = transFun
        self.p = p # how often to train on new whale

    def __len__(self):
        return self.Ids.shape[0]

    def __getitem__(self, index):
        r = np.random.rand()
        if r < self.p:
            indexes = [self.newWhale[int(r*self.n_newWhale)]]
        else:
            img_list = self.Ids.iloc[index]['Imgs']
            np.random.shuffle(img_list)
            indexes = img_list[:2]
        return self.__data_generation_color(indexes)

    def __data_generation_color(self, indexes):
        imgs_list = [np.load(img) for img in indexes]
        if len(imgs_list)==1:
            return self.transFun(imgs_list[0]).transpose(2,0,1),self.transFun(imgs_list[0]).transpose(2,0,1)
        else:
            return self.transFun(imgs_list[0]).transpose(2,0,1),self.transFun(imgs_list[1]).transpose(2,0,1)
        
class PredictGenerator():
    # used for TTA prediction
    # not supported for dataloader
    def __init__(self, Ids, transFun, TTASize, keepOrg=True):
        self.Ids = Ids # a list of lists, like [[w1_img1,w1_img2...],[w2_img1,w2_img2...],...]
        self.transFun = transFun
        self.TTASize = TTASize
        self.keepOrg = keepOrg
        
    def __len__(self):
        'Denotes the number of batches per epoch.'
        return len(self.Ids)

    def __getitem__(self, index):
        'Generate one batch of data'
        if index < len(self):
            indexes = self.__create2(self.Ids[index])
            X = self.__data_generation_color(indexes)
            return X
        else:
            raise IndexError

    def __create2(self,img_list):
        len_ = len(img_list)
        np.random.shuffle(img_list)
        if len_ <= self.TTASize:
            return img_list
        else:
            return img_list[:self.TTASize]
        
    def __data_generation_color(self, indexes):
        imgs_list = [np.load(img) for img in indexes]
        size_ = len(indexes)
        imgs_list = [img_ if ((j < size_) and self.keepOrg) else self.transFun(img_)
                     for j,img_ in enumerate(imgs_list * (self.TTASize//size_) + imgs_list[:self.TTASize%size_])]
        return torch.from_numpy(np.array(imgs_list).transpose(0,3,1,2))
    
class SpatialTransformerNet(nn.Module):
    def __init__(self,localization,fc_loc,size=None):
        # locNet should be a model that has 6 outputs for affine transformation
        # size should be something like torch.Size([1, 3, 256, 256])
        super(SpatialTransformerNet, self).__init__()
        self.localization = localization 
        self.fc_loc = fc_loc
        self.size = size
        
        # Spatial transformer localization-network
#        self.localization = nn.Sequential(
#            nn.Conv2d(1, 8, kernel_size=7),
#            nn.MaxPool2d(2, stride=2),
#            nn.ReLU(True),
#            nn.Conv2d(8, 10, kernel_size=5),
#            nn.MaxPool2d(2, stride=2),
#            nn.ReLU(True)
#        )
#
#        # Regressor for the 3 * 2 affine matrix
#        self.fc_loc = nn.Sequential(
#            nn.Linear(10 * 3 * 3, 32),
#            nn.ReLU(True),
#            nn.Linear(32, 3 * 2)
#        )

        # Initialize the weights/bias with identity transformation
#        self.fc_loc[2].weight.data.zero_()
#        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_loc[0].in_features)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size() if self.size is None else self.size)
        x = F.grid_sample(x, grid)
        return x

class SpatialTransformerNetMutiple(nn.Module):
    def __init__(self,localization,fc_loc,multiple,size=None):
        # locNet should be a model that has 6 outputs for affine transformation
        # size should be a list without batch [3, 256, 256]
        # fc_loc should output 2*3*mutiple
        # all share the same convBase
        # return a list of Xs
        super(SpatialTransformerNetMutiple, self).__init__()
        self.localization = localization 
        self.fc_loc = fc_loc
        self.size = size
        self.multiple = multiple
        
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_loc[0].in_features)
        thetas = self.fc_loc(xs)
        thetas = [thetas[:,i::self.multiple].view(-1, 2, 3) for i in range(self.multiple)]
        grids = [F.affine_grid(theta, x.size() if self.size is None else \
                               torch.Size([x.size()[0]]+self.size)) for theta in thetas]
        return [F.grid_sample(x, grid) for grid in grids]
    
class STN_multiple_model(nn.Module):
    def __init__(self,stn,convNet):
        # convNet can either be a list of convNet, one for each multiple or 
        # one convNet shared by all multiple. ConvNet should return shape (N,D)
        super(STN_multiple_model, self).__init__()
        self.stn = stn 
        self.convNet = convNet
        
    def forward(self, x):
        x_tsf = self.stn(x)
        if isinstance(self.convNet,list):
            return torch.cat([net(x) for x,net in zip(x_tsf,self.convNet)],1)
        else:
            return torch.cat([self.convNet(x) for x in x_tsf],1)
    
    
l2_distance = lambda x1,x2: torch.mean((x1-x2)**2,1)
l2_distance_np = lambda x1,x2: np.mean((x1-x2)**2,3)

def loss_func_generator(HalfBatch,margin,distanceFun):
    def loss_func(model,data):
        X1,X2 = data
        feature_anchor = model(X1)
        feature_pos = model(X2)
        r = torch.randint(1,HalfBatch,(1,))[0]
        index = torch.fmod(torch.arange(HalfBatch)+r,HalfBatch)
        feature_neg = feature_pos[index]
        d_pos = distanceFun(feature_anchor,feature_pos)
        d_neg = distanceFun(feature_anchor,feature_neg)
        margin_torch = torch.ones_like(d_pos,device='cuda:0')*margin
        triplet_loss = torch.mean(torch.max(d_pos-d_neg,margin_torch))
        return triplet_loss
    return loss_func

def top_k(d,k=5,returnValue=False):
    top = np.argpartition(d,k)[0:k]
    index = np.argsort(d[top])
    if returnValue:
        return top[index].tolist(),d[top[index]].tolist()
    else:
        return top[index].tolist()

def loop_distance(feature_train,feature_val,distanceFun,aggFun,returnValue=False,k=5):
	# aggFun is an aggregating function like partial(np.quantile,q=0.5,axis=(1,2))
    feature_train = feature_train[:,:,np.newaxis,:]
    index_list = []
    if returnValue:
        value_list = []

    for feature in feature_val:
        feature = feature[np.newaxis,np.newaxis,:,:]
        d = aggFun(distanceFun(feature_train,feature))
        if returnValue:
            index,value = top_k(d,k=k,returnValue=returnValue)
            index_list.append(index)
            value_list.append(value)
        else:
            index = top_k(d,k=k,returnValue=returnValue)
            index_list.append(index)

    if returnValue:
        return np.array(index_list),np.array(value_list)
    else:
        return np.array(index_list)

def MAP(labels,predicts):
    # labels are 1-d array, predicts are 2-d array (number_of_imgs,k_guesses)
    # does not work if predicts have repeats in row
    _,temp = np.where(predicts == labels[:,np.newaxis])
    return np.sum(1/(temp + 1))/labels.shape[0]


    
    
