#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:56:22 2018

@author: will
"""
import tensorflow as tf
import numpy as np
from utility.grayscale_imagenet import Xception_greyscale
from tensorflow.keras import models,layers
from tensorflow.keras import optimizers


'''
# test augmentation
from utility.plotting import aug_compare
for i in range(10):
    for img in Ids_train.iloc[i]['Imgs']:
        img = np.load(img)
        aug_compare(img, transform,cmap='gray')
'''

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, Ids, newWhale, transFun, shuffle=True,HalfBatch=16):
        self.Ids = Ids # df with first column being w ID, second being list of imgs for each w
        self.newWhale = newWhale # a list of img for new whale
        self.transFun = transFun
        self.shuffle = shuffle
        self.HalfBatch = HalfBatch
        self.y = np.ones(HalfBatch*2,dtype=np.float32) 
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
        return [X1,X2], self.y

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
        imgs_list = [[self.transFun(group[0])[:,:,np.newaxis],self.transFun(group[0])[:,:,np.newaxis]] if len(group)==1 
                      else [self.transFun(group[0])[:,:,np.newaxis],self.transFun(group[1])[:,:,np.newaxis]] for group in imgs_list]
        X1,X2 = list(zip(*imgs_list))
        r = np.random.randint(0,self.HalfBatch)
        X1,X2 = list(X1),list(X2)
        X1.extend(X1)
        X2.extend([X2[(i+r)%self.HalfBatch] for i in range(16)])
        return np.array(X1),np.array(X2)
    
    
''' build model '''    
conv_base = Xception_greyscale((256,256,1),'max',False)
feature_model = models.Sequential()
feature_model.add(conv_base)
#feature_model.add(layers.Dense(1024,activation=tf.sigmoid))
feature_model.add(layers.Dense(1024))

img1 = layers.Input(shape=(256,256,1))
img2 = layers.Input(shape=(256,256,1))
feature1 = feature_model(img1)
feature2 = feature_model(img2)

#output = layers.Lambda(lambda features: tf.reduce_mean(features[0]*features[1],axis=1,keepdims=True))([feature1,feature2])
output = layers.Lambda(lambda features: tf.sigmoid(tf.reduce_mean(features[0]*features[1],axis=1,keepdims=True)))([feature1,feature2])
model = models.Model([img1,img2],output)

def loss_fun_factory(margin_p=0.95,margin_n=0.1):
    # margin_p for positive, margin_n for negative examples
    def loss(y_true,y_pred):
        return -1 * y_true * tf.minimum(y_pred,margin_p) + (1-y_true) * tf.maximum(y_pred,margin_n)
    return loss

loss_ = loss_fun_factory()
model.compile(loss=loss_,
optimizer=optimizers.Adam(lr=1e-3),
metrics=[loss_])



''' set up generators '''

import pickle
from albumentations import ShiftScaleRotate,Cutout,RandomContrast,RandomBrightness,Compose
from utility.albumentations_helper import create_transform
import time

with open('/home/will/Desktop/kaggle/Whale/train_df.pkl', 'rb') as f:
    Ids_train = pickle.load(f)
with open('/home/will/Desktop/kaggle/Whale/new_whale_train.pkl', 'rb') as f:
    newWhale_train = pickle.load(f)
with open('/home/will/Desktop/kaggle/Whale/val_df.pkl', 'rb') as f:
    Ids_val = pickle.load(f)
with open('/home/will/Desktop/kaggle/Whale/new_whale_val.pkl', 'rb') as f:
    newWhale_val = pickle.load(f)

aug = Compose([RandomContrast(p=0.2),RandomBrightness(p=0.2),
                ShiftScaleRotate(shift_limit=0.03,rotate_limit=15,scale_limit=0.02,p=1),Cutout(p=0.5)])
transform = create_transform(aug)    

gen_train = DataGenerator(Ids_train,newWhale_train,transform)
gen_val = DataGenerator(Ids_val,newWhale_val,transform)


''' train model '''


'''reset weights
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import backend as K
session = K.get_session()
dense_layer = model.layers[2].layers[1]
dense_layer.set_weights([glorot_uniform()(dense_layer.get_weights()[0].shape).eval(session=session),
                         np.zeros_like(dense_layer.get_weights()[1],dtype=np.float32)])
#K.clear_session()
'''

start = time.time()
history = model.fit_generator(
          gen_train,
          validation_data = gen_val,
          epochs=5,
          use_multiprocessing=True,workers=3,max_queue_size=20)
end = time.time()
print('time:{}'.format(end - start))


