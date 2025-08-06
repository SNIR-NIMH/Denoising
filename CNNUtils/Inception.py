#from __future__ import print_function, division

#import tensorflow as tf # don't import tf within a def, it will cause model saving error when trained with multiple gpu
from keras import Input, Model
#from keras.layers.advanced_activations import PReLU, ReLU
from keras.layers import PReLU, ReLU
from keras.layers import (AveragePooling2D, AveragePooling3D, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, concatenate,
                          Lambda, Conv2DTranspose, Conv3DTranspose, UpSampling3D, UpSampling2D, BatchNormalization, Activation)
import keras.backend as K
K.set_image_data_format('channels_last')
from keras.layers import add



def inception_block(input, base_filters):
    conv_inception1a = Conv2D(base_filters * 2, (1, 1), activation='relu', padding='same')(input)

    conv_inception2a = Conv2D(base_filters * 3, (1, 1), activation='relu', padding='same')(input)
    conv_inception4a = Conv2D(base_filters * 4, (3, 3), activation='relu', padding='same')(conv_inception2a)

    conv_inception3a = Conv2D(base_filters, (1, 1), activation='relu', padding='same')(input)
    conv_inception5a = Conv2D(base_filters , (5, 5), activation='relu', padding='same')(conv_inception3a)

    # pool_inception1a = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inlayer)
    # conv_inception6a = Conv2D(base_filters * 2, (1, 1), activation='relu', padding='same')(pool_inception1a)

    pool_inception2a = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
    conv_inception7a = Conv2D(base_filters , (1, 1), activation='relu', padding='same')(pool_inception2a)

    outlayer = concatenate([conv_inception1a, conv_inception4a], axis=-1)
    outlayer = concatenate([outlayer, conv_inception5a], axis=-1)
    # outlayer = concatenate([outlayer, conv_inception6a], axis=-1)
    outlayer = concatenate([outlayer, conv_inception7a], axis=-1)
    return outlayer

def inception_block3d(input, base_filters):
    conv_inception1a = Conv3D(base_filters * 2, (1, 1,1), activation='relu', padding='same')(input)

    conv_inception2a = Conv3D(base_filters * 3, (1, 1,1), activation='relu', padding='same')(input)
    conv_inception4a = Conv3D(base_filters * 4, (3, 3,3), activation='relu', padding='same')(conv_inception2a)

    conv_inception3a = Conv3D(base_filters, (1, 1,1), activation='relu', padding='same')(input)
    conv_inception5a = Conv3D(base_filters , (5, 5,5), activation='relu', padding='same')(conv_inception3a)

    # pool_inception1a = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inlayer)
    # conv_inception6a = Conv2D(base_filters * 2, (1, 1), activation='relu', padding='same')(pool_inception1a)

    pool_inception2a = MaxPooling3D(pool_size=(3, 3,3), strides=(1, 1,1), padding='same')(input)
    conv_inception7a = Conv3D(base_filters , (1, 1,1), activation='relu', padding='same')(pool_inception2a)

    outlayer = concatenate([conv_inception1a, conv_inception4a], axis=-1)
    outlayer = concatenate([outlayer, conv_inception5a], axis=-1)
    # outlayer = concatenate([outlayer, conv_inception6a], axis=-1)
    outlayer = concatenate([outlayer, conv_inception7a], axis=-1)
    return outlayer

def Inception2D(input, numfilter):  # UNET-ish version of Inception

    conv1d = inception_block(input, numfilter)

    pool1d = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1d)

    conv2d = inception_block(pool1d, numfilter*2)

    pool2d = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2d)

    conv3d = inception_block(pool2d, numfilter*4)

    pool3d = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3d)

    conv4d = inception_block(pool3d,numfilter*8)

    pool4d = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4d)

    conv5d = inception_block(pool4d, numfilter*16)

    up4u = UpSampling2D(size=(2, 2))(conv5d)

    conv4u = concatenate([conv4d, up4u], axis=-1)

    conv4u = inception_block(conv4u, numfilter*8)

    up3u = UpSampling2D(size=(2, 2))(conv4u)

    conv3u = concatenate([up3u, conv3d], axis=-1)
    conv3u = inception_block(conv3u, numfilter*4)

    up2u = UpSampling2D(size=(2, 2))(conv3u)

    conv2u = concatenate([up2u, conv2d], axis=-1)
    conv2u = inception_block(conv2u, numfilter*2)

    up1u = UpSampling2D(size=(2, 2))(conv2u)

    conv1u = concatenate([up1u, conv1d], axis=-1)
    conv1u = inception_block(conv1u, numfilter)

    final = Conv2D(numfilter, (3, 3), activation='relu', padding='same', strides=(1, 1))(conv1u)
    return  final


def Inception3D(input, numfilter):  # UNET-ish version of Inception

    conv1d = inception_block3d(input, numfilter)

    pool1d = MaxPooling3D(pool_size=(2, 2,2), strides=(2, 2,2), padding='same')(conv1d)

    conv2d = inception_block3d(pool1d, numfilter*2)

    pool2d = MaxPooling3D(pool_size=(2, 2,2), strides=(2, 2, 2), padding='same')(conv2d)

    conv3d = inception_block3d(pool2d, numfilter*4)

    pool3d = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(conv3d)

    conv4d = inception_block3d(pool3d,numfilter*6)

    pool4d = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(conv4d)

    conv5d = inception_block3d(pool4d, numfilter*8)

    up4u = UpSampling3D(size=(2, 2, 2))(conv5d)

    conv4u = concatenate([conv4d, up4u], axis=-1)

    conv4u = inception_block3d(conv4u, numfilter*6)

    up3u = UpSampling3D(size=(2, 2, 2))(conv4u)

    conv3u = concatenate([up3u, conv3d], axis=-1)
    conv3u = inception_block3d(conv3u, numfilter*4)

    up2u = UpSampling3D(size=(2, 2, 2))(conv3u)

    conv2u = concatenate([up2u, conv2d], axis=-1)
    conv2u = inception_block3d(conv2u, numfilter*2)

    up1u = UpSampling3D(size=(2, 2, 2))(conv2u)

    conv1u = concatenate([up1u, conv1d], axis=-1)
    conv1u = inception_block3d(conv1u, numfilter)

    final = Conv3D(numfilter, (3, 3, 3), activation='relu', padding='same', strides=(1, 1,1))(conv1u)
    return  final



'''

def Inception2d(inlayer, base_filters):
    conv_inception1a = Conv2D(base_filters * 4, (1, 1), activation='relu', padding='same')(inlayer)

    conv_inception2a = Conv2D(base_filters * 6, (1, 1), activation='relu', padding='same')(inlayer)
    conv_inception4a = Conv2D(base_filters * 8, (3, 3), activation='relu', padding='same')(conv_inception2a)

    conv_inception3a = Conv2D(base_filters, (1, 1), activation='relu', padding='same')(inlayer)
    conv_inception5a = Conv2D(base_filters * 2, (5, 5), activation='relu', padding='same')(conv_inception3a)

    #pool_inception1a = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inlayer)
    #conv_inception6a = Conv2D(base_filters * 2, (1, 1), activation='relu', padding='same')(pool_inception1a)

    pool_inception2a = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inlayer)
    conv_inception7a = Conv2D(base_filters * 2, (1, 1), activation='relu', padding='same')(pool_inception2a)

    outlayer = concatenate([conv_inception1a, conv_inception4a], axis=-1)
    outlayer = concatenate([outlayer, conv_inception5a], axis=-1)
    #outlayer = concatenate([outlayer, conv_inception6a], axis=-1)
    outlayer = concatenate([outlayer, conv_inception7a], axis=-1)
    return outlayer


def Inception3d(inlayer, base_filters):
    conv_inception1a = Conv3D(base_filters * 2, (1, 1, 1), activation='relu', padding='same')(inlayer)

    conv_inception2a = Conv3D(base_filters * 3, (1, 1, 1), activation='relu', padding='same')(inlayer)
    conv_inception4a = Conv3D(base_filters * 4, (3, 3, 3), activation='relu', padding='same')(conv_inception2a)

    conv_inception3a = Conv3D(base_filters, (1, 1, 1), activation='relu', padding='same')(inlayer)
    conv_inception5a = Conv3D(base_filters * 2, (5, 5, 5), activation='relu', padding='same')(conv_inception3a)

    pool_inception1a = AveragePooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inlayer)
    conv_inception6a = Conv3D(base_filters * 2, (1, 1, 1), activation='relu', padding='same')(pool_inception1a)

    pool_inception2a = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inlayer)
    conv_inception7a = Conv3D(base_filters * 2, (1, 1, 1), activation='relu', padding='same')(pool_inception2a)

    outlayer = concatenate([conv_inception1a, conv_inception4a], axis=-1)
    outlayer = concatenate([outlayer, conv_inception5a], axis=-1)
    outlayer = concatenate([outlayer, conv_inception6a], axis=-1)
    outlayer = concatenate([outlayer, conv_inception7a], axis=-1)
    return outlayer
'''