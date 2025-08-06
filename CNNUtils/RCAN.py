import tensorflow as tf
#from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Lambda, GlobalAveragePooling2D, Multiply, Dense, Reshape
#from tensorflow.keras.models import Model
from keras.layers import Input, Conv2D, Conv3D, Activation, Add, Lambda, GlobalAveragePooling2D, GlobalAveragePooling3D,\
    Multiply, Dense, Reshape
from keras import backend as K

K.set_image_data_format('channels_last')


import sys
sys.setrecursionlimit(10000)


def get_spatial_ndim(x):
    return K.ndim(x) - 2  # batchsize x H x W x D x C or batchsize x H x W x C

'''

def sub_pixel_conv2d(scale=2, **kwargs):
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)


def upsample(input_tensor, filters):
    x = Conv2D(filters=filters * 4, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = sub_pixel_conv2d(scale=2)(x)
    x = Activation('relu')(x)
    return x
'''

def ca(input_tensor, filters, reduce=8, dim=2):
    if dim==2:
        x = GlobalAveragePooling2D()(input_tensor)
    else:
        x = GlobalAveragePooling3D()(input_tensor)
    x = Reshape((1, 1, filters))(x)
    x = Dense(filters//reduce,  activation='relu', kernel_initializer='he_normal', use_bias=False)(x)
    x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)
    x = Multiply()([x, input_tensor])
    return x


def rcab(input_tensor, filters, scale=0.1, dim=2):
    if dim==2:
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
    else:
        x = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = Activation('relu')(x)
    if dim==2:
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    else:
        x = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = ca(x, filters, reduce=8, dim=dim)
    if scale:
        x = Lambda(lambda t: t * scale)(x)
    x = Add()([x, input_tensor])

    return x

# n_rcab more than 6 gives instability for 2D training
def rg(input_tensor, filters, n_rcab=5, dim=2):
    x = input_tensor
    if dim==3:
        n_rcab = 3

    for _ in range(n_rcab):
        x = rcab(x, filters, dim=dim)
    if dim==2:
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    else:
        x = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x, input_tensor])

    return x

# n_rcab more than 6 gives instability for 2D training, and more than 3 gives instability for 3D training
def rir(input_tensor, filters, n_rg=5, dim=2):
    x = input_tensor
    if dim ==3:
        n_rg = 3
    for _ in range(n_rg):
        x = rg(x, filters=filters, dim=dim)
    if dim==2:
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    else:
        x = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x, input_tensor])

    return x


'''
def generator(filters=64, n_sub_block=2):
    inputs = Input(shape=(None, None, 3))

    x = x_1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)
    x = rir(x, filters=filters)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x_1, x])

    for _ in range(n_sub_block):
        x = upsample(x, filters)
    x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)

    return Model(inputs=inputs, outputs=x)
'''

def RCAN(inputs, filters):

    n_sub_block=2
    dim = get_spatial_ndim(inputs)
    if dim!= 2 and dim!= 3:
        sys.exit('ERROR: Patch size must be 2D or 3D.')

    if dim==2:
        x = x_1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)
    else:
        x = x_1 = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)

    x = rir(x, filters=filters, dim=dim)

    if dim==2:
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    else:
        x = Conv3D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    x = Add()([x_1, x])


    '''
    for _ in range(n_sub_block):
        x = upsample(x, filters)
        print(x.shape)
    '''
    if dim==2:
        x = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(x)
    else:
        x = Conv3D(filters=1, kernel_size=3, strides=1, padding='same')(x)

    return x
    #return Model(inputs=inputs, outputs=x)