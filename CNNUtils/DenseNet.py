from __future__ import print_function, division

import tensorflow as tf # don't import tf within a def, it will cause model saving error when trained with multiple gpu
from keras.engine import Input, Model
from keras.layers.advanced_activations import PReLU, ReLU
from keras.layers import (AveragePooling2D, AveragePooling3D, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, concatenate,
                          Lambda, Conv2DTranspose, Conv3DTranspose, UpSampling3D, UpSampling2D, BatchNormalization,
                          Activation, SpatialDropout3D, SpatialDropout2D)

from keras.layers.merge import add, concatenate
import keras.backend as K
K.set_image_data_format('channels_last')


def layer3D(x, nb_feature_maps, working_axis=-1, kernel=(3, 3, 3)):
    bn = BatchNormalization(axis=working_axis)(x)
    relu = Activation("relu")(bn)
    conv = Conv3D(nb_feature_maps, kernel, padding="same", strides=(1, 1, 1), kernel_initializer="he_uniform")(relu)
    drop = SpatialDropout3D(0.5)(conv)
    return drop

def layer2D(x, nb_feature_maps, working_axis=-1, kernel=(3, 3)):
    bn = BatchNormalization(axis=working_axis)(x)
    relu = Activation("relu")(bn)
    conv = Conv2D(nb_feature_maps, kernel, padding="same", strides=(1, 1), kernel_initializer="he_uniform")(relu)
    drop = SpatialDropout2D(0.5)(conv)
    return drop


def dense_block3D(x, steps, nb_feature_maps, working_axis=-1):
    connections = []
    x_stack = x
    for i in range(steps):
        l = layer3D(x_stack, nb_feature_maps, working_axis=working_axis)
        connections.append(l)
        x_stack = concatenate([x_stack, l], axis=working_axis)
    return x_stack, connections

def dense_block2D(x, steps, nb_feature_maps, working_axis=-1):
    connections = []
    x_stack = x
    for i in range(steps):
        l = layer2D(x_stack, nb_feature_maps, working_axis=working_axis)
        connections.append(l)
        x_stack = concatenate([x_stack, l], axis=working_axis)
    return x_stack, connections

def transition_down3D(x, nb_feature_maps=16, working_axis=-1):
    l = layer3D(x, nb_feature_maps, working_axis, (1, 1, 1))
    l = MaxPooling3D((2, 2, 2))(l)
    return l

def transition_down2D(x, nb_feature_maps=16, working_axis=-1):
    l = layer2D(x, nb_feature_maps, working_axis, (1, 1))
    l = MaxPooling2D((2, 2))(l)
    return l


def transition_up3D(skip, blocks, nb_feature_maps=16, working_axis=-1):
    l = concatenate(blocks,axis=-1)
    l = Conv3DTranspose(nb_feature_maps, (3, 3, 3), strides=(2, 2, 2), padding="same", kernel_initializer="he_uniform")(l)
    l = concatenate([l, skip], axis=-1)
    return l

def transition_up2D(skip, blocks, nb_feature_maps=16, working_axis=-1):
    l = concatenate(blocks,axis=-1)
    l = Conv2DTranspose(nb_feature_maps, (3, 3), strides=(2, 2), padding="same", kernel_initializer="he_uniform")(l)
    l = concatenate([l, skip], axis=-1)
    return l


def DenseNet3D(input, basefilters):

    growth_rate = basefilters #16
    #steps = [4, 5, 7, 10, 12]
    #last_step = 5 #15
    steps = [1, 3, 5, 7]
    last_step = 4 #15

    stack = Conv3D(basefilters, (3, 3, 3), padding="same", strides=(1,1,1), kernel_initializer="he_uniform")(input)

    # Encode part
    skip_connection_list = []
    for s in steps:
        # Dense Block
        stack, _ = dense_block3D(stack, s, growth_rate, working_axis=-1)
        skip_connection_list.append(stack)
        # Transition Down
        stack = transition_down3D(stack, stack._keras_shape[-1])

    skip_connection_list = skip_connection_list[::-1]

    # Encoded filtering
    block_to_upsample = []
    for i in range(last_step):
        l = layer3D(stack, growth_rate, working_axis=-1)
        block_to_upsample.append(l)
        stack = concatenate([stack, l],axis=-1)

    # Decode path
    x_stack = stack
    x_block_to_upsample = block_to_upsample
    n_layers_per_block = [last_step, ] + steps[::-1]
    for n_layers, s, skip in zip(n_layers_per_block, steps[::-1], skip_connection_list):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers
        x_stack = transition_up3D(skip, x_block_to_upsample, n_filters_keep)
        # Dense Block
        x_stack, x_block_to_upsample = dense_block3D(x_stack, s, growth_rate, working_axis=-1)

    # output layers
    out = Conv3D(basefilters, (1, 1, 1), kernel_initializer="he_uniform", padding="same")(x_stack)
    return out



def DenseNet2D(input, basefilters):

    growth_rate = basefilters #16
    #steps = [4, 5, 7, 10, 12]
    #last_step = 5 #15
    steps = [1, 3, 5, 7]
    last_step = 4 #15

    stack = Conv2D(basefilters, (3, 3), padding="same", strides=(1,1), kernel_initializer="he_uniform")(input)

    # Encode part
    skip_connection_list = []
    for s in steps:
        # Dense Block
        stack, _ = dense_block2D(stack, s, growth_rate, working_axis=-1)
        skip_connection_list.append(stack)
        # Transition Down
        stack = transition_down2D(stack, stack._keras_shape[-1])

    skip_connection_list = skip_connection_list[::-1]

    # Encoded filtering
    block_to_upsample = []
    for i in range(last_step):
        l = layer2D(stack, growth_rate, working_axis=-1)
        block_to_upsample.append(l)
        stack = concatenate([stack, l],axis=-1)

    # Decode path
    x_stack = stack
    x_block_to_upsample = block_to_upsample
    n_layers_per_block = [last_step, ] + steps[::-1]
    for n_layers, s, skip in zip(n_layers_per_block, steps[::-1], skip_connection_list):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers
        x_stack = transition_up2D(skip, x_block_to_upsample, n_filters_keep)
        # Dense Block
        x_stack, x_block_to_upsample = dense_block2D(x_stack, s, growth_rate, working_axis=-1)

    # output layers
    out = Conv2D(basefilters, (1, 1), kernel_initializer="he_uniform", padding="same")(x_stack)
    return out