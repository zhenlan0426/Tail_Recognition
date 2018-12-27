#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:56:22 2018

@author: will
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import models,layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras import backend as K

'''
# test augmentation
from utility.plotting import aug_compare
for i in range(10):
    for img in Ids_train.iloc[i]['Imgs']:
        img = np.load(img)
        aug_compare(img, transform,cmap='gray')
'''

class FineTune(Callback):
  """Reduce learning rate and set more layers trainable to True when a metric has stopped improving.

  ```
  Arguments:
      monitor: quantity to be monitored.
      factor: factor by which the learning rate will
          be reduced. new_lr = lr * factor
      patience: number of epochs with no improvement
          after which learning rate will be reduced.
      verbose: int. 0: quiet, 1: update messages.
      fine_tune_layerName: list of layer name to be turn on one after another
      e.g. ['conv5','conv4','conv3']
      save_model is path to save models to. should be a list of two for train_model and feature_model
  """

  def __init__(self,
               fine_tune_layerName,
               lossFun,
               optimizer,
               monitor='val_loss',
               save_model=None,
               save_model_after=10,
               factor=0.25,
               patience=5,
               verbose=1):
    super(FineTune, self).__init__()

    self.monitor = monitor
    self.factor = factor
    self.patience = patience
    self.verbose = verbose
    self.wait = 0
    self.best = np.Inf
    self.save_model = save_model
    self.save_model_after = save_model_after
    self.fine_tune_layerName = fine_tune_layerName
    self.fine_tune_layer = 0
    self.lossFun = lossFun
    self.optimizer = optimizer

  def on_epoch_end(self, epoch, logs=None):
      # turns on part of the network and reduce lr
      logs = logs or {}
      logs['lr'] = K.get_value(self.model.optimizer.lr)
      current = logs[self.monitor]
      if current < self.best:
          self.best = current
          self.wait = 0
          if (self.save_model is not None) and epoch > self.save_model_after:
              self.model.save(self.save_model[0])
              self.model.layers[2].layers[0].save(self.save_model[1])

      else:
          self.wait += 1
          if self.wait >= self.patience:
              if self.fine_tune_layer >= len(self.fine_tune_layerName):
                  print('stoped training at epoch:{}'.format(epoch))
                  self.model.stop_training = True
              else:   
                  if self.verbose > 0:
                      print('reduce lr at epoch:{}'.format(epoch))
                      
                  ## custom fine-tune ##
                  self.model.layers[2].layers[0].trainable=True
                  train = False
                  for layer in self.model.layers[2].layers[0].layers:
                      if self.fine_tune_layerName[self.fine_tune_layer] in layer.name:
                          train = True
                      layer.trainable = train
                  self.model.compile(loss=self.lossFun,optimizer=self.optimizer(lr=logs['lr']*self.factor))
                  ## custom fine-tune ##
                  
                  self.wait = 0
                  self.fine_tune_layer += 1

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, Ids, newWhale, transFun, color, shuffle=True,HalfBatch=16):
        self.Ids = Ids # df with first column being w ID, second being list of imgs for each w
        self.newWhale = newWhale # a list of img for new whale
        self.transFun = transFun
        self.color = color
        self.shuffle = shuffle
        self.HalfBatch = HalfBatch
        self.y = np.ones(HalfBatch*2,dtype=np.float32) 
        self.y[HalfBatch:] = 0 # 16 positive, 16 negative examples
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch. subtract one as one example from newWhale'
        return int(self.Ids.shape[0] / (self.HalfBatch-1)) - 1

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.Ids.iloc[index*(self.HalfBatch-1):(index+1)*(self.HalfBatch-1)]['Imgs'].tolist()
        indexes.append([self.newWhale[index]])
        X1,X2 = self.__data_generation_color([self.__create2(i) for i in indexes]) if self.color else \
                self.__data_generation([self.__create2(i) for i in indexes])
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
        r = np.random.randint(1,self.HalfBatch)
        X1,X2 = list(X1),list(X2)
        X1.extend(X1)
        X2.extend([X2[(i+r)%self.HalfBatch] for i in range(self.HalfBatch)])
        return np.array(X1),np.array(X2)
    
    def __data_generation_color(self, indexes):
        imgs_list = [[np.load(img) for img in group] for group in indexes]
        imgs_list = [[self.transFun(group[0]),self.transFun(group[0])] if len(group)==1 
                      else [self.transFun(group[0]),self.transFun(group[1])] for group in imgs_list]
        X1,X2 = list(zip(*imgs_list))
        r = np.random.randint(1,self.HalfBatch)
        X1,X2 = list(X1),list(X2)
        X1.extend(X1)
        X2.extend([X2[(i+r)%self.HalfBatch] for i in range(self.HalfBatch)])
        return np.array(X1),np.array(X2)
    
class PredictGenerator(tf.keras.utils.Sequence):
    # used for TTA prediction
    def __init__(self, Ids, transFun, TTASize, color):
        self.Ids = Ids # a list of lists, like [[w1_img1,w1_img2...],[w2_img1,w2_img2...],...]
        self.transFun = transFun
        self.TTASize = TTASize
        self.color = color
        
    def __len__(self):
        'Denotes the number of batches per epoch.'
        return len(self.Ids)

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.__create2(self.Ids[index])
        X = self.__data_generation_color(indexes) if self.color else self.__data_generation(indexes)
        return X

    def __create2(self,img_list):
        len_ = len(img_list)
        np.random.shuffle(img_list)
        if len_ <= self.TTASize:
            return img_list
        else:
            return img_list[:self.TTASize]
        
    def __data_generation(self, indexes):
        imgs_list = [np.load(img) for img in indexes]
        size_ = len(indexes)
        imgs_list = [self.transFun(img_)[:,:,np.newaxis] if j >= size_ else img_[:,:,np.newaxis]
                     for j,img_ in enumerate(imgs_list * (self.TTASize//size_) + imgs_list[:self.TTASize%size_])]
        return np.array(imgs_list)
    
    def __data_generation_color(self, indexes):
        imgs_list = [np.load(img) for img in indexes]
        size_ = len(indexes)
        imgs_list = [self.transFun(img_) if j >= size_ else img_
                     for j,img_ in enumerate(imgs_list * (self.TTASize//size_) + imgs_list[:self.TTASize%size_])]
        return np.array(imgs_list)
    
''' tfs '''    
def l2_distance(feature1,feature2):
    return tf.reduce_mean(tf.squared_difference(feature1,feature2),axis=1,keepdims=True)

def sigmoid_dot_distance(feature1,feature2):
    return -1*tf.sigmoid(tf.reduce_mean(feature1*feature2,axis=1,keepdims=True))

def dot_distance(feature1,feature2):
    return -1*tf.reduce_mean(feature1*feature2,axis=1,keepdims=True)

def dot_distance_neg(feature1,feature2):
    return tf.reduce_mean(feature1*feature2,axis=1,keepdims=True)

def dot_sigmoid_distance(feature1,feature2):
    return -1*tf.reduce_mean(tf.sigmoid(feature1)*tf.sigmoid(feature2),axis=1,keepdims=True)


def margin_loss_fun_factory(margin_p,margin_n):
    # margin_p for positive, margin_n for negative examples
    def loss(y_true,y_pred):
        return y_true * tf.maximum(y_pred,margin_p) - (1-y_true) * tf.minimum(y_pred,margin_n)
    return loss

def exp_loss(y_true,y_pred):
    return tf.exp(y_true*y_pred-(1-y_true)*y_pred)

def cross_entropy_loss(y_true,y_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)

def create_model(lr,distanceFun,lossFun,conv_base,IsColor,nodes=[512],activations=[None]):
    ''' distanceFun takes feature1,feature2 as inputs and returns their 'distance'
        conv_base is something like Xception_greyscale((256,256,1),'max',False)
        IsColor then 3 channels else 1
    '''
    feature_model = models.Sequential()
    feature_model.add(conv_base)
    for i,act in zip(nodes,activations):
        feature_model.add(layers.Dense(i,activation=act))

    img1 = layers.Input(shape=(224,224,3 if IsColor else 1))
    img2 = layers.Input(shape=(224,224,3 if IsColor else 1))
    feature1 = feature_model(img1)
    feature2 = feature_model(img2)

    output = layers.Lambda(lambda features: distanceFun(features[0],features[1]))([feature1,feature2])
    train_model = models.Model([img1,img2],output)
    feature_model.compile(loss='mse',optimizer='sgd') # needed to run predict_gen
    train_model.compile(loss=lossFun,optimizer=optimizers.Adam(lr=lr))
    return train_model,feature_model

def create_model2(lr,lossFun,conv_base,IsColor):

    img1 = layers.Input(shape=(224,224,3 if IsColor else 1))
    img2 = layers.Input(shape=(224,224,3 if IsColor else 1))
    feature1 = conv_base(img1)
    feature2 = conv_base(img2)
    
    x1 = layers.Lambda(lambda x : x[0]*x[1])([feature1, feature2])
    x2 = layers.Lambda(lambda x : tf.abs(x[0] - x[1]))([feature1, feature2])
    x3 = layers.Lambda(lambda x : tf.square(x[0] - x[1]))([feature1, feature2])
    x4 = layers.Lambda(lambda x : x[0] + x[1])([feature1, feature2])
    x = layers.Concatenate()([x1, x2, x3, x4])
    x = layers.Reshape((4, 1, conv_base.output_shape[1]), name='reshape1')(x)
    output = layers.DepthwiseConv2D(kernel_size=(4,1),use_bias=False)(x)
    output = layers.Lambda(lambda x:tf.squeeze(tf.reduce_mean(x,3,keepdims=True),(1,2)))(output)

    train_model = models.Model([img1,img2],output)
    conv_base.compile(loss='mse',optimizer='sgd') # needed to run predict_gen
    train_model.compile(loss=lossFun,optimizer=optimizers.Adam(lr=lr))
    return train_model,conv_base

def Xception_reduced(input_shape):
    input_ = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3),
                      strides=(2, 2),
                      use_bias=False,
                      name='block1_conv1')(input_)
    x = layers.BatchNormalization(name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv1')(x)
    x = layers.BatchNormalization(name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = layers.BatchNormalization(name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv1')(x)
    x = layers.BatchNormalization(name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = layers.BatchNormalization(name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(728, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(x)
    x = layers.BatchNormalization(name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv2')(x)
    x = layers.BatchNormalization(name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    x = layers.add([x, residual])
    x = layers.GlobalMaxPooling2D()(x)
    model = models.Model(input_, x, name='xception')
    return model

'''reset weights
from tensorflow.keras.initializers import glorot_uniform
session = K.get_session()
dense_layer = model.layers[2].layers[1]
dense_layer.set_weights([glorot_uniform()(dense_layer.get_weights()[0].shape).eval(session=session),
                         np.zeros_like(dense_layer.get_weights()[1],dtype=np.float32)])

'''


''' predictions '''  
def generate_feature(Ids,transform,FFA_size,color,feature_model):
    feature_gen = PredictGenerator(Ids if isinstance(Ids,list) else Ids.Imgs.tolist(),transform,FFA_size,color)
    feature = feature_model.predict_generator(feature_gen,workers=2,use_multiprocessing=True)
    return np.reshape(feature,(feature.shape[0]//FFA_size,FFA_size,feature.shape[1]))

def l2_distance_np(feature1,feature2):
    return np.mean((feature1-feature2)**2,3)

def dot_distance_neg_np(feature1,feature2):
    return -np.mean(feature1*feature2,axis=3)

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

