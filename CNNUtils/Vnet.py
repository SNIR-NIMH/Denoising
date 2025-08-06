from __future__ import print_function, division

import tensorflow as tf # don't import tf within a def, it will cause model saving error when trained with multiple gpu
from keras.engine import Input, Model
from keras.layers.advanced_activations import PReLU, ReLU
from keras.layers import (AveragePooling2D, AveragePooling3D, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, concatenate,
                          Lambda, Conv2DTranspose, Conv3DTranspose, UpSampling3D, UpSampling2D)

from keras.layers.merge import add
import keras.backend as K
K.set_image_data_format('channels_last')

def Vnet3D(input, base_filters):
    # Ideally all ReLUs should be PreLUs, but keras 2.2.4 does not support NoneType for PreLU,
    # so replacing all PreLUs by ReLUs

    conv1 = ReLU()(Conv3D(base_filters, (3, 3, 3), strides=(1, 1, 1), activation=None, padding='same')(input))

    conv1d = ReLU()(
        Conv3D(base_filters * 2, kernel_size=(3, 3, 3), activation=None, strides=(2, 2, 2), padding='same')(conv1))

    conv2 = ReLU()(
        Conv3D(base_filters * 2, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv1d))
    conv2 = ReLU()(Conv3D(base_filters * 2, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv2))
    conv2 = add([conv2, conv1d])

    conv2d = ReLU()(
        Conv3D(base_filters * 4, kernel_size=(2, 2, 2), activation=None, strides=(2, 2, 2), padding='same')(conv2))

    conv3 = ReLU()(
        Conv3D(base_filters * 4, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv2d))
    conv3 = ReLU()(Conv3D(base_filters * 4, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv3))
    conv3 = ReLU()(Conv3D(base_filters * 4, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv3))
    conv3 = add([conv3, conv2d])

    conv3d = ReLU()(
        Conv3D(base_filters * 8, kernel_size=(2, 2, 2), activation=None, strides=(2, 2, 2), padding='same')(conv3))

    conv4 = ReLU()(
        Conv3D(base_filters * 8, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv3d))
    conv4 = ReLU()(Conv3D(base_filters * 8, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv4))
    conv4 = ReLU()(Conv3D(base_filters * 8, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv4))
    conv4 = add([conv4, conv3d])

    conv4d = ReLU()(
        Conv3D(base_filters * 16, kernel_size=(2, 2, 2), activation=None, strides=(2, 2, 2), padding='same')(conv4))

    conv5 = ReLU()(
        Conv3D(base_filters * 16, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv4d))
    conv5 = ReLU()(
        Conv3D(base_filters * 16, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv5))
    conv5 = ReLU()(
        Conv3D(base_filters * 16, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv5))
    conv5 = add([conv5, conv4d])

    conv5u = ReLU()(
        Conv3DTranspose(base_filters * 8, kernel_size=(2, 2, 2), activation=None, strides=(2, 2, 2), padding='same')(conv5))
    conv6 = concatenate([conv5u, conv4])

    conv6 = ReLU()(Conv3D(base_filters * 8, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv6))
    conv6 = ReLU()(Conv3D(base_filters * 8, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv6))
    conv6 = ReLU()(Conv3D(base_filters * 8, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv6))
    conv6 = add([conv6, conv5u])

    conv6u = ReLU()(
        Conv3DTranspose(base_filters * 4, kernel_size=(2, 2, 2), activation=None, strides=(2, 2, 2), padding='same')(conv6))
    conv7 = concatenate([conv6u, conv3])

    conv7 = ReLU()(Conv3D(base_filters * 4, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv7))
    conv7 = ReLU()(Conv3D(base_filters * 4, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv7))
    conv7 = ReLU()(Conv3D(base_filters * 4, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv7))
    conv7 = add([conv7, conv6u])

    conv7u = ReLU()(
        Conv3DTranspose(base_filters * 2, kernel_size=(2, 2, 2), activation=None, strides=(2, 2, 2), padding='same')(conv7))
    conv8 = concatenate([conv7u, conv2])

    conv8 = ReLU()(Conv3D(base_filters * 2, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv8))
    conv8 = ReLU()(Conv3D(base_filters * 2, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv8))
    conv8 = add([conv8, conv7u])

    conv8u = ReLU()(
        Conv3DTranspose(base_filters, kernel_size=(2, 2, 2), activation=None, strides=(2, 2, 2), padding='same')(conv8))
    conv9 = concatenate([conv8u, conv1])

    conv9 = ReLU()(Conv3D(base_filters, kernel_size=(3, 3, 3), activation=None, strides=(1, 1, 1), padding='same')(conv9))
    final = add([conv9, conv8u])

    return final


def Vnet2D(input, base_filters):
    # Ideally all ReLUs should be PreLUs, but keras 2.2.4 does not support NoneType for PreLU,
    # so replacing all PreLUs by ReLUs

    conv1 = ReLU()(Conv2D(base_filters, (3,3), strides=(1,1), activation=None, padding='same')(input))

    conv1d = ReLU()(Conv2D(base_filters *2 , kernel_size=(3,3), activation=None, strides=(2,2), padding='same')(conv1))

    conv2 = ReLU()(Conv2D(base_filters * 2, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv1d))
    conv2 = ReLU()(Conv2D(base_filters * 2, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv2))
    conv2 = add([conv2, conv1d])

    conv2d = ReLU()(Conv2D(base_filters * 4, kernel_size=(2, 2), activation=None, strides=(2, 2), padding='same')(conv2))

    conv3 = ReLU()(Conv2D(base_filters * 4, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv2d))
    conv3 = ReLU()(Conv2D(base_filters * 4, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv3))
    conv3 = ReLU()(Conv2D(base_filters * 4, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv3))
    conv3 = add([conv3, conv2d])

    conv3d = ReLU()(Conv2D(base_filters * 8, kernel_size=(2, 2), activation=None, strides=(2, 2), padding='same')(conv3))

    conv4 = ReLU()(Conv2D(base_filters * 8, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv3d))
    conv4 = ReLU()(Conv2D(base_filters * 8, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv4))
    conv4 = ReLU()(Conv2D(base_filters * 8, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv4))
    conv4 = add([conv4, conv3d])

    conv4d = ReLU()(Conv2D(base_filters * 16, kernel_size=(2, 2), activation=None, strides=(2, 2), padding='same')(conv4))

    conv5 = ReLU()(Conv2D(base_filters * 16, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv4d))
    conv5 = ReLU()(Conv2D(base_filters * 16, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv5))
    conv5 = ReLU()(Conv2D(base_filters * 16, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv5))
    conv5 = add([conv5, conv4d])

    conv5u = ReLU()(Conv2DTranspose(base_filters * 8, kernel_size=(2,2), activation=None, strides=(2,2), padding='same')(conv5))
    conv6 = concatenate([conv5u, conv4])

    conv6 = ReLU()(Conv2D(base_filters * 8, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv6))
    conv6 = ReLU()(Conv2D(base_filters * 8, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv6))
    conv6 = ReLU()(Conv2D(base_filters * 8, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv6))
    conv6 = add([conv6, conv5u])

    conv6u = ReLU()(Conv2DTranspose(base_filters * 4, kernel_size=(2, 2), activation=None, strides=(2, 2), padding='same')(conv6))
    conv7 = concatenate([conv6u, conv3])

    conv7 = ReLU()(Conv2D(base_filters * 4, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv7))
    conv7 = ReLU()(Conv2D(base_filters * 4, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv7))
    conv7 = ReLU()(Conv2D(base_filters * 4, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv7))
    conv7 = add([conv7, conv6u])

    conv7u = ReLU()(Conv2DTranspose(base_filters * 2, kernel_size=(2, 2), activation=None, strides=(2, 2), padding='same')(conv7))
    conv8 = concatenate([conv7u, conv2])

    conv8 = ReLU()(Conv2D(base_filters * 2, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv8))
    conv8 = ReLU()(Conv2D(base_filters * 2, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv8))
    conv8 = add([conv8, conv7u])

    conv8u = ReLU()(Conv2DTranspose(base_filters, kernel_size=(2, 2), activation=None, strides=(2, 2), padding='same')(conv8))
    conv9 = concatenate([conv8u, conv1])

    conv9 = ReLU()(Conv2D(base_filters, kernel_size=(3, 3), activation=None, strides=(1, 1), padding='same')(conv9))
    final = add([conv9, conv8u])

    return final
