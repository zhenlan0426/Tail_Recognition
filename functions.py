#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:56:22 2018

@author: will
"""
import tensorflow as tf
import numpy as np


import pickle
with open('/home/will/Desktop/kaggle/Whale/train_df.pkl', 'rb') as f:
    Ids = pickle.load(f)
with open('/home/will/Desktop/kaggle/Whale/new_whale_val.pkl', 'rb') as f:
    newWhale = pickle.load(f)


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, Ids, newWhale, transFun, shuffle=True,HalfBatch=16):
        self.Ids = Ids # df with first column being w ID, second being list of imgs for each w
        self.newWhale = newWhale # a list of img for new whale
        self.transFun = transFun
        self.shuffle = shuffle
        self.HalfBatch = HalfBatch
        self.y = np.ones(HalfBatch*2,dtype=np.int8) 
        self.y[HalfBatch:] = 0 # 16 positive, 16 negative examples
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch. subtract one as one example from newWhale'
        return int(self.Ids.shape[0] / (self.HalfBatch-1))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.Ids.iloc[index*(self.HalfBatch-1):(index+1)*(self.HalfBatch-1)]['Imgs'].tolist()
        indexes.append([self.newWhale[index]])
        X1,X2 = self.__data_generation([self.__create2(i) for i in indexes])
        return (X1,X2), self.y

    def on_epoch_end(self):
        if self.shuffle == True:
            self.Ids = self.Ids.sample(frac=1).reset_index(drop=True)
            np.random.shuffle(self.newWhale)
            
    @staticmethod
    def __create2(img_list):
        len_ = len(img_list)
        if len_ <= 2:
            return img_list
        else:
            np.random.shuffle(img_list)
            return img_list[:2]
        
    def __data_generation(self, indexes):
        imgs_list = [[np.load(img) for img in group] for group in indexes]
        imgs_list = [[self.transFun(group[0]),self.transFun(group[0])] if len(group)==1 
                      else [self.transFun(group[0]),self.transFun(group[1])] for group in imgs_list]
        X1,X2 = list(zip(*imgs_list))
        r = np.random.randint(0,self.HalfBatch)
        X1,X2 = list(X1),list(X2)
        X1.extend(X1)
        X2.extend([X2[(i+r)%self.HalfBatch] for i in range(16)])
        return np.array(X1),np.array(X2)
    
    
    
