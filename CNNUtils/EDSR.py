import os, sys
import numpy as np
import  tensorflow as tf
from keras.engine import Input, Model
from keras.layers.advanced_activations import PReLU, ReLU
from keras import regularizers
from keras.layers import (AveragePooling2D, AveragePooling3D, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, concatenate,
                          Lambda, Conv2DTranspose, Conv3DTranspose, UpSampling3D, UpSampling2D)

from keras.layers.merge import add
import keras.backend as K


def res_block2D(input_tensor, numfilter, res_scale=1.0):
    x = Conv2D(numfilter, (3, 3), padding='same', activation='relu',
               activity_regularizer=regularizers.l1(10e-10))(input_tensor)
    x = Conv2D(numfilter, (3, 3), padding='same', activity_regularizer=regularizers.l1(10e-10))(x)
    x = Lambda(lambda x: x * res_scale)(x)
    x = add([x, input_tensor])
    return x

def res_block3D(input_tensor, numfilter, res_scale=1.0):
    x = Conv3D(numfilter, (3, 3, 3), padding='same', activation='relu',
               activity_regularizer=regularizers.l1(10e-10))(input_tensor)
    x = Conv3D(numfilter, (3, 3, 3), padding='same', activity_regularizer=regularizers.l1(10e-10))(x)
    x = Lambda(lambda x: x * res_scale)(x)
    x = add([x, input_tensor])
    return x

def EDSR2D(inputs, numfilter):

    n_resblocks = 32
    x = Conv2D(numfilter, (3,3), padding='same', activity_regularizer=regularizers.l1(10e-10))(inputs)
    res_scale = 0.1  # scaling by 1 is a poor choice, safer is scaling by 0.1
    conv1  = x
    for i in range(n_resblocks):
        x = res_block2D(x, numfilter, res_scale)

    x = Conv2D(numfilter, 3, padding='same', activity_regularizer=regularizers.l1(10e-10))(x)
    x = add([x, conv1])

    return x


def EDSR3D(inputs, numfilter):
    n_resblocks = 8
    x = Conv3D(numfilter, (3, 3, 3), padding='same', activity_regularizer=regularizers.l1(10e-10))(inputs)
    res_scale = 0.1  # scaling by 1 is a poor choice, safer is scaling by 0.1
    conv1 = x
    for i in range(n_resblocks):
        x = res_block3D(x, numfilter, res_scale)

    x = Conv3D(numfilter, (3,3,3), padding='same', activity_regularizer=regularizers.l1(10e-10))(x)
    x = add([x, conv1])

    return x