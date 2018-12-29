#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 12:42:40 2018

@author: will
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import models,layers
from tensorflow.keras import optimizers

def create_model__(lr,lossFun,conv_base,IsColor):
    feature1 = layers.Input(shape=conv_base.output_shape[1:])
    feature2 = layers.Input(shape=conv_base.output_shape[1:])
    x1 = layers.Lambda(lambda x : x[0]*x[1])([feature1, feature2])
    x2 = layers.Lambda(lambda x : tf.abs(x[0] - x[1]))([feature1, feature2])
    x3 = layers.Lambda(lambda x : tf.square(x[0] - x[1]))([feature1, feature2])
    x4 = layers.Lambda(lambda x : x[0] + x[1])([feature1, feature2])
    x = layers.Concatenate()([x1, x2, x3, x4])
    x = layers.Reshape((4, 1, conv_base.output_shape[1]), name='reshape1')(x)
    output = layers.DepthwiseConv2D(kernel_size=(4,1),use_bias=False)(x)
    output = layers.Lambda(lambda x:tf.squeeze(tf.reduce_mean(x,3,keepdims=True),(1,2)))(output)
    head_model = models.Model([feature1,feature2],output)
    
    img1 = layers.Input(shape=(224,224,3 if IsColor else 1))
    img2 = layers.Input(shape=(224,224,3 if IsColor else 1))
    y = head_model([conv_base(img1),conv_base(img2)])
    train_model = models.Model([img1,img2],y)
    
    conv_base.compile(loss='mse',optimizer='sgd') # needed to run predict_gen
    head_model.compile(loss='mse',optimizer='sgd') # needed to run predict_gen    
    train_model.compile(loss=lossFun,optimizer=optimizers.Adam(lr=lr))
    return train_model,conv_base,head_model

class FeaturesGenerator(tf.keras.utils.Sequence):
    # self.N1 needs to be divisible by batchSize!!!
    def __init__(self, feature1,feature2,batchSize):
        # feature1 are numpy array of shape (N,d)
        self.feature1 = feature1 
        self.feature2 = feature2
        self.batchSize = batchSize
        self.N1 = self.feature1.shape[0]
        self.N2 = self.feature2.shape[0]
        self.N = self.N1 * self.N2
        self.len_ = self.N1//self.batchSize
        self.d = self.feature1.shape[1]
        
    def __len__(self):
        'Denotes the number of batches per epoch.'
        return self.N//self.batchSize

    def __getitem__(self, index):
        'Generate one batch of data'
        i = index%self.len_
        j = index//self.len_
        X1 = self.feature1[i*self.batchSize:(i+1)*self.batchSize]
        X2 = np.broadcast_to(self.feature2[j],(self.batchSize,self.d))
        return [X1,X2]