import nibabel as nifti
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import random
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
import copy
import tempfile
from tqdm import tqdm
import numpy as np
from scipy import ndimage
from scipy.signal import argrelextrema
import argparse
import h5py
#from keras.utils.multi_gpu_utils import multi_gpu_model
#from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
#from tensorflow import keras
#from keras.utils.multi_gpu_utils import multi_gpu_model
#from keras_contrib.losses import  dssim

from tensorflow import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import  Input, Model
from keras.layers import (Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, AveragePooling2D,
                          AveragePooling3D, concatenate, Lambda, Conv2DTranspose, UpSampling3D,
                          UpSampling2D)
from keras.optimizers import Adam
from keras.utils import  Sequence
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.models import load_model
from keras.utils.layer_utils import print_summary
import statsmodels.api as sm
import time
from skimage import io
from sklearn.mixture import GaussianMixture
from pynvml import *


path = os.path.dirname(sys.argv[0])
path=os.path.abspath(path)
path=os.path.join(path,'CNNUtils')
print('Appending {}'.format(path))
sys.path.append(path)

#from CNNUtils.multi_gpu_utils import multi_gpu_model
from CNNUtils.Inception import Inception2D, Inception3D
from CNNUtils.Unet import Unet2D,Unet3D
from CNNUtils.UnetPlusPlus import UnetPlusPlus2D
#from CNNUtils.Vnet import Vnet2D,Vnet3D
#from CNNUtils.EDSR import EDSR2D, EDSR3D
from CNNUtils.RCAN import RCAN
from CNNUtils.AttentionUnet import AttentionUnet2D
#from CNNUtils.DenseNet import  DenseNet2D, DenseNet3D
#from CNNUtils.FPN import FPN2D, FPN3D
from CNNUtils.utils import normalize_image, pad_image, Crop, pad_image
from CNNUtils.utils import dice_coeff, dice_coeff_loss, focal_loss, read_image, ssim_loss, ssim_metric


def shuff_array(imgpatch, targetpatch, dim=2):
    N=imgpatch.shape[0]
    rng = random.SystemRandom()
    list=np.asarray(range(0,N), dtype=int)
    np.random.shuffle(list)
    if dim==2:
        imgpatch = imgpatch[list, :, :, :]
        targetpatch = targetpatch[list, :, :, :]
    else:
        imgpatch = imgpatch[list, :, :, :, :]
        targetpatch = targetpatch[list, :, :, :, :]
    return  imgpatch, targetpatch

def GetPatches(Image4D, Target, mask, opt, ImagePatches, TargPatches, atlasid):

    rng = random.SystemRandom()
    patchsize = opt['patchsize']
    patchsize = np.asarray(patchsize, dtype=int)
    dsize = patchsize//2
    nummodal = len(opt['modalities']) -1
    # All non-zero voxels in the target are used to search for valid patches.
    # Clearly, this is not the best way to find all valid patches. Ideally
    # there should be a user-input mask from where the patches are chosen.
    if mask is not None:
        indx = np.nonzero(mask)
    else:
        indx = np.nonzero(Target > 0)
    indx=np.asarray(indx,dtype=np.int16) # This won't work if the image dimensions >65535. This is alright since
    # training images are generally small, but this is sitll dangerous


    L=len(indx[0])
    L1=np.minimum(opt['maxpatch'],L)
    print("Number of patches used  = %d (out of %d, max %d, %.2f %%)" %(L1,L,opt['maxpatch'],100.0*L1/L))
    randindx=rng.sample(range(0, L),L1)
    newindx=np.ndarray((3,L1))


    for i in range(0,L1):
        for j in range(0,3):
            newindx[j,i]=indx[j,randindx[i]]

    newindx=np.asarray(newindx,dtype=int)

    count = L1 * (atlasid)
    if len(patchsize) == 3:
        for i in range(0,L1):
            I = newindx[0,i]
            J = newindx[1,i]
            K = newindx[2,i]
            for t in range(0,nummodal):
                ImagePatches[count+i, :, :, :, t] = Image4D[I - dsize[0]:I + dsize[0], J - dsize[1]:J + dsize[1],
                                              K - dsize[2]:K + dsize[2], t ]
            TargPatches[count+i, :, :, :, 0] = Target[I - dsize[0]:I + dsize[0] , J - dsize[1]:J + dsize[1] ,
                                         K - dsize[2]:K + dsize[2] ]
    else:
        for i in range(0,L1):
            I = newindx[0,i]
            J = newindx[1,i]
            K = newindx[2,i]
            for t in range(0,nummodal):
                ImagePatches[count+i, :, :, t] = Image4D[I - dsize[0]:I + dsize[0], J - dsize[1]:J + dsize[1], K, t ]
            TargPatches[count+i, :, :,0] = Target[I - dsize[0]:I + dsize[0], J - dsize[1]:J + dsize[1], K ]


    #return ImagePatches,TargPatches




def get_model_3d(opt):
    base_filters = opt['basefilter']
    numchannel = int(len(opt['modalities'])) - 1

    if opt['numgpu'] == 1:
        input = Input((None, None, None, int(numchannel)))
        if opt['model'] == 'unet':
            final = Unet3D(input, base_filters)
        elif opt['model'] == 'inception':
            final = Inception3D (input, base_filters)
            #final = Inception3d(final, base_filters)
            #final = Inception3d(final, base_filters)
        elif opt['model'] == 'densenet':
            final = DenseNet3D(input, base_filters)
        elif opt['model'] == 'rcan':
            final = RCAN(input, base_filters)
        else:
            sys.exit('ERROR: Model must be Inception or UNET. You entered %s' %(opt['model']))

        if opt['loss'] == 'bce':
            final = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', strides=(1, 1, 1))(final)
        else:
            final = Conv3D(1, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(final)

        model = Model(inputs=input, outputs=final)
        #print(model.summary())

        #if opt['numgpu'] > 1:
        #    model = ModelMGPU(model, gpus=opt['numgpu'])

        if opt['loss'] == 'bce':
            model.compile(optimizer=Adam(learning_rate=opt['lr']), loss='binary_crossentropy', metrics=['accuracy'])
        elif opt['loss'] == 'ssim':
            model.compile(optimizer=Adam(learning_rate=opt['lr']), loss=ssim_loss, metrics=[ssim_metric])
        elif opt['loss'] == 'mae':
            model.compile(optimizer=Adam(learning_rate=opt['lr']), loss='mae')
        elif opt['loss'] == 'mse':
            model.compile(optimizer=Adam(learning_rate=opt['lr']), loss='mse')

    else:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            input = Input((None, None, None, int(numchannel)))
            if opt['model'] == 'unet':
                final = Unet3D(input, base_filters)
            elif opt['model'] == 'inception':
                final = Inception3D(input, base_filters)
                #final = Inception3d(final, base_filters)
                #final = Inception3d(final, base_filters)
            elif opt['model'] == 'densenet':
                final = DenseNet3D(input, base_filters)
            elif opt['model'] == 'rcan':
                final = RCAN(input, base_filters)
            else:
                sys.exit('ERROR: Model must be Inception or UNET. You entered %s' % (opt['model']))

            if opt['loss'] == 'bce':
                final = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', strides=(1, 1, 1))(final)
            else:
                final = Conv3D(1, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(final)

            model = Model(inputs=input, outputs=final)
            # print(model.summary())

            #if opt['numgpu'] > 1:
            #    model = ModelMGPU(model, gpus=opt['numgpu'])

            if opt['loss'] == 'bce':
                model.compile(optimizer=Adam(learning_rate=opt['lr']), loss='binary_crossentropy', metrics=['accuracy'])
            elif opt['loss'] == 'ssim':
                model.compile(optimizer=Adam(learning_rate=opt['lr']), loss=ssim_loss, metrics=[ssim_metric])
            elif opt['loss'] == 'mae':
                model.compile(optimizer=Adam(learning_rate=opt['lr']), loss='mae')
            elif opt['loss'] == 'mse':
                model.compile(optimizer=Adam(learning_rate=opt['lr']), loss='mse')
    return model


def get_model_2d(opt):
    base_filters = opt['basefilter']
    numchannel = int(len(opt['modalities'])) - 1


    if opt['numgpu'] == 1:
        input = Input((None, None, int(numchannel)))
        if opt['model'] == 'unet':
            final = Unet2D(input, base_filters)
        elif opt['model'] == 'attentionunet':
            final = AttentionUnet2D(input, base_filters)
        elif opt['model'] == 'inception':
            final = Inception2D(input, base_filters)
        elif opt['model'] == 'densenet':
            final = DenseNet2D(input, base_filters)
        elif opt['model'] == 'edsr':
            final = EDSR2D(input, base_filters)
        elif opt['model'] == 'rcan':
            final = RCAN(input, base_filters)
        elif opt['model'] == 'unet++':
            final = UnetPlusPlus2D(input, base_filters, deep_supervision=False)
        else:
            sys.exit('ERROR: Model must be Inception or UNET. You entered %s' % (opt['model']))
        if opt['loss'] == 'bce':
            final = Conv2D(1, (3, 3), activation='sigmoid', padding='same', strides=(1, 1))(final)
        else:
            final = Conv2D(1, (3, 3), activation='relu', padding='same', strides=(1, 1))(final)
        model = Model(inputs=input, outputs=final)
        if opt['loss'] == 'bce':
            model.compile(optimizer=Adam(learning_rate=opt['lr']), loss='binary_crossentropy', metrics=['accuracy'])
        elif opt['loss'] == 'ssim':
            model.compile(optimizer=Adam(learning_rate=opt['lr']), loss=ssim_loss, metrics=[ssim_metric])
        elif opt['loss'] == 'mae':
            model.compile(optimizer=Adam(learning_rate=opt['lr']), loss='mae')
        elif opt['loss'] == 'mse':
            model.compile(optimizer=Adam(learning_rate=opt['lr']), loss='mse')
    else:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            # Everything must be withing the strategy's scope
            input = Input((None, None, int(numchannel)))
            if opt['model'] == 'unet':
                final = Unet2D(input, base_filters)
            elif opt['model'] == 'attentionunet':
                final = AttentionUnet2D(input, base_filters)
            elif opt['model'] == 'inception':
                final = Inception2D(input, base_filters)
            elif opt['model'] == 'densenet':
                final = DenseNet2D(input, base_filters)
            elif opt['model'] == 'edsr':
                final = EDSR2D(input, base_filters)
            elif opt['model'] == 'rcan':
                final = RCAN(input, base_filters)
            elif opt['model'] == 'unet++':
                final = UnetPlusPlus2D(input, base_filters, deep_supervision=False)
            else:
                sys.exit('ERROR: Model must be Inception or UNET. You entered %s' % (opt['model']))
            if opt['loss'] == 'bce':
                final = Conv2D(1, (3, 3), activation='sigmoid', padding='same', strides=(1, 1))(final)
            else:
                final = Conv2D(1, (3, 3), activation='relu', padding='same', strides=(1, 1))(final)

            model = Model(inputs=input, outputs=final)
            if opt['loss'] == 'bce':
                model.compile(optimizer=Adam(learning_rate=opt['lr']), loss='binary_crossentropy', metrics=['accuracy'])
            elif opt['loss'] == 'ssim':
                model.compile(optimizer=Adam(learning_rate=opt['lr']), loss=ssim_loss, metrics=[ssim_metric])
            elif opt['loss'] == 'mae':
                model.compile(optimizer=Adam(learning_rate=opt['lr']), loss='mae')
            elif opt['loss'] == 'mse':
                model.compile(optimizer=Adam(learning_rate=opt['lr']), loss='mse')

    #print(model.summary())
    #print(model.count_params())


    return model

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size, :, :,:]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size,:,:,: ]
        return batch_x, batch_y



def main(opt):

    numatlas = opt['numatlas']
    numgpu = opt['numgpu']
    patchsize = opt['patchsize']
    modalities = opt['modalities']
    patchsize = np.asarray(patchsize,dtype=int)
    if len(patchsize) ==2:
        padsize = np.max(patchsize)//2 + 1 # 2D patching ==> patch only in xy by the max(patchsize)
    else:
        padsize = patchsize//2 + 1 # 3D patching ==> padsize = half of patchsize
    nummodal = int(len(modalities)) - 1 # Number of input modalities to use, e.g. atlas1_M1, atlas1_M2,..atlas1_Mx.
    time_id = time.strftime('%d-%m-%Y_%H-%M-%S')
    uid = str(os.path.basename(tempfile.mktemp())[3:].upper())
    print('Unique ID is %s %s' % (uid,time_id))
    con = '+'.join([str(mod).upper() for mod in opt['modalities']])
    psize = 'x'.join([str(side) for side in opt['patchsize']])
    modelname = 'DenoiseModel_' + uid + '_' + time_id + '_' + psize + '_' + opt['model'].upper() + '_' + con + '.h5'
    modelname = os.path.join(opt['outdir'],modelname)
    print("Model will be written at %s" % (modelname))

    if len(patchsize)==3:
        matsize = (opt['maxpatch']*numatlas, patchsize[0],patchsize[1],patchsize[2],nummodal)
        matsize2 = (opt['maxpatch'] * numatlas, patchsize[0], patchsize[1], patchsize[2], 1)
    else:
        matsize = (opt['maxpatch'] * numatlas, patchsize[0], patchsize[1], nummodal)
        matsize2 = (opt['maxpatch'] * numatlas, patchsize[0], patchsize[1], 1)

    print('Approximate size of training data = %.1f GB' % (
            2 * np.prod(patchsize) * (nummodal + 1) * 4.0 * opt['maxpatch'] * numatlas / (1024.0 ** 3)))
    ImagePatches=np.empty(matsize,dtype=np.float32)
    TargetPatches=np.empty(matsize2,dtype=np.float32)

    patch_count = 0
    for i in range(0,numatlas):
        targ = None
        s = 'atlas' + str(i + 1) + '_GT'
        s = os.path.join(atlasdir, s)
        for ext in ['.nii','.nii.gz','.h5','.tif','.tiff']:
            s1 = s + ext
            if os.path.isfile(s1):
                targ = read_image(s1)
                break


        if targ is None:
            sys.exit('ERROR: Ground truth image %s must be either nii, nii.gz, tif, or h5.' %(s))
        else:
            targ = np.asarray(targ, dtype=np.float32)

        targ,_ = normalize_image(targ, modalities[-1])

        targ = pad_image(targ,padsize, len(patchsize))

        dim = targ.shape
        print('Padded image size {}'.format(dim))
        dim = np.append(dim,nummodal)

        vol4d = np.zeros(dim,dtype=np.float32)


        mask = None
        s = 'atlas' + str(i + 1) + '_mask'
        s = os.path.join(atlasdir, s)
        for ext in ['.nii','.nii.gz','.h5','.tif','.tiff']:
            s1 = s + ext
            if os.path.isfile(s1):
                mask = read_image(s1)
                break

        if mask is None:
            print('WARNING: Mask file can not be read. I will choose patches from all over the image.')
        else:
            mask = np.asarray(mask, dtype=np.float32)
            mask = pad_image(mask,padsize, len(patchsize))


        for j in range(0,nummodal):

            s = 'atlas' + str(i + 1) + '_M' + str(j + 1)
            s = os.path.join(atlasdir, s)
            for ext in ['.nii', '.nii.gz', '.h5', '.tif', '.tiff']:
                s1 = s + ext
                if os.path.isfile(s1):
                    temp = read_image(s1)
                    break

            if temp is None:
                sys.exit('Image %s must be either nii, nii.gz, tif, or h5' %(s))
            else:
                temp = np.asarray(temp,dtype=np.float32)
            temp = pad_image(temp,padsize, len(patchsize))
            temp,_ = normalize_image(temp,modalities[j])

            vol4d[:,:,:,j] = np.asarray(temp,dtype=np.float32)

        print("Image size after padding = %s & %s" %(str(vol4d.shape), str(targ.shape)))

        GetPatches(vol4d, targ, mask, opt, ImagePatches, TargetPatches, i) # assuming maxpatches is always less than
                                                                        # total number of available patches in the image

        '''
        if str(opt['model']).lower() == 'rcan':
            ImagePatches = ImagePatches/65535 # scale between 0 and 1
            TargetPatches = TargetPatches/65535
        '''

        print('-' * 100)

    print("Size of the input matrices are " + str(ImagePatches.shape) + " and " + str(TargetPatches.shape))
    #print(ImagePatches.dtype)
    #print(TargetPatches.dtype)
    if opt['loss'] == 'bce':
        tempoutname = modelname.replace('.h5', '_epoch-{epoch:03d}_val-acc-{val_acc:.4f}.h5')
        callbacks = [ModelCheckpoint(tempoutname, monitor='val_acc', verbose=1, save_best_only=True,
                                 save_freq='epoch', mode='max')]
        dlr = ReduceLROnPlateau(monitor="val_acc", factor=0.5, patience=10, min_delta=0.0001,
                                mode='max', verbose=1, cooldown=2, min_lr=1e-8)
        earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=10,
                              verbose=1, mode='max')
    elif opt['loss'] == 'ssim':
        tempoutname = modelname.replace('.h5', '_epoch-{epoch:03d}_val-ssim-{val_ssim_metric:.4f}.h5')
        callbacks = [ModelCheckpoint(tempoutname, monitor='val_ssim_metric', verbose=1, save_best_only=True,
                                     save_freq='epoch', mode='max')]
        dlr = ReduceLROnPlateau(monitor="val_ssim_metric", factor=0.5, patience=10, min_delta=0.0001,
                                mode='max', verbose=1, cooldown=2, min_lr=1e-8)
        earlystop = EarlyStopping(monitor='val_ssim_metric', min_delta=0.0001, patience=10,
                                  verbose=1, mode='max')
    else:
        tempoutname = modelname.replace('.h5', '_epoch-{epoch:03d}_val-loss-{val_loss:.4f}.h5')
        callbacks = [ModelCheckpoint(tempoutname, monitor='val_loss', verbose=1, save_best_only=True,
                                     save_freq='epoch', mode='min')]
        dlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_delta=0.0001,
                                mode='min', verbose=1, cooldown=2, min_lr=1e-8)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10,
                                  verbose=1, mode='min')

    callbacks.append(dlr)
    callbacks.append(earlystop)

    #print(np.max(np.ndarray.flatten(ImagePatches)))
    #print(np.max(np.ndarray.flatten(TargetPatches)))

    if len(patchsize) == 3:
        model = get_model_3d(opt)
        if opt['initmodel'] != 'None' and os.path.exists(opt['initmodel']):
            dict = {"tf": tf,
                    #"ssim_metric": ssim_metric,
                    #"ssim_loss": ssim_loss,
                    }

            try:
                oldmodel = load_model(opt['initmodel'], custom_objects=dict)
                model.set_weights(oldmodel.get_weights())
                print("Initializing from existing model %s" % (opt['initmodel']))
            except Exception as e:
                print('WARNING: Can not load from initial model.')
                print(str(e))
        print("Total number of parameters = {:,}".format(model.count_params()))
        ImagePatches, TargetPatches = shuff_array(ImagePatches, TargetPatches, dim=3)
        d1 = int(ImagePatches.shape[0])
        d2 = int(np.ceil(d1 * 0.80))

        train_gen = DataGenerator(ImagePatches[0:d2, :, :, :, :], TargetPatches[0:d2, :, :, :, :], opt['batchsize'])
        test_gen = DataGenerator(ImagePatches[d2:d1, :, :, :, :], TargetPatches[d2:d1, :, :, :, :], opt['batchsize'])

        history = model.fit(train_gen, epochs=opt['epoch'], batch_size=opt['batchsize'],
                            callbacks=callbacks, verbose=1, validation_data=test_gen)
        #model.fit(ImagePatches, TargetPatches, batch_size=opt['batchsize'], epochs=opt['epoch'], verbose=1,
        #          validation_split=0.2, callbacks=callbacks, shuffle=True)
    else:
        model = get_model_2d(opt)
        if opt['initmodel'] != 'None' and os.path.exists(opt['initmodel']):
            dict = {"tf": tf,
                    #"dice_coeff": dice,
                    #"ssim_loss": ssim_loss,
                    }
            try:
                oldmodel = load_model(opt['initmodel'],custom_objects=dict)
                model.set_weights(oldmodel.get_weights())
                print("Initializing from existing model %s" % (opt['initmodel']))
            except Exception as e:
                print('Can not load from initial model.')
                print(str(e))
        print("Total number of parameters = {:,}".format(model.count_params()))
        ImagePatches, TargetPatches = shuff_array(ImagePatches, TargetPatches, dim=2)
        d1=int(ImagePatches.shape[0])
        d2=int(np.ceil(d1*0.80))

        train_gen = DataGenerator(ImagePatches[0:d2,:,:,:], TargetPatches[0:d2,:,:,:], opt['batchsize'])
        test_gen = DataGenerator(ImagePatches[d2:d1,:,:,:], TargetPatches[d2:d1,:,:,:], opt['batchsize'])

        history = model.fit(train_gen, epochs=opt['epoch'], batch_size=opt['batchsize'],
                            callbacks=callbacks, verbose=1, validation_data=test_gen)


        #model.fit(ImagePatches, TargetPatches, batch_size=opt['batchsize'], epochs=opt['epoch'], verbose=1,
        #          validation_split=0.2, callbacks=callbacks, shuffle=True)

    print("Model is written at " + modelname)
    if numgpu > 1:
        opt2 = copy.deepcopy(opt)
        opt2['numgpu'] = 1
        with tf.device("/cpu:0"):
            if len(patchsize) == 2:
                single_model = get_model_2d(opt2)
            else:
                single_model = get_model_3d(opt2)

        single_model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy')
        single_model.set_weights(model.get_weights())
        single_model.save(filepath=modelname)
    else:
        model.save(filepath=modelname)





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Image Denoising Training.', formatter_class=argparse.RawDescriptionHelpFormatter)
    required = parser.add_argument_group('Required arguments')
    required.add_argument('--atlasdir', required=True, action='store', dest='ATLASDIR', type=str,
                        help='Atlas directory containing atlasXX_M1.nii.gz, atlasXX_M2.nii.gz, atlasXX_GT.nii.gz, XX=1,2,3.. etc. '
                            'The M1, M2 etc denote 1st, 2nd modalities and GT denotes ground truth. See --modalities. '
                            'The atlases should be devoid of any background noise, because by default all patches with non-zero '
                            'center voxels will be considered for training. If the atlases have background noise, '
                            'then the noise patches may be included in training. Therefore use the '
                            'remove_background_noise.sh script to remove background noise from one of the T1-w images. '
                            'The brainmask can easily be obtained using ROBEX. Alternately, if atlasXX_mask.nii.gz '
                            'binary images are present, then patches will be collected from the non-zero indices of '
                             'the atlasXX_mask.nii.gz images.')
    required.add_argument('--natlas',required=True, action='store', type=int, dest='NUMATLAS',
                        help='Number of atlases to be used. Atlas directory must contain at least '
                             'these many atlases. Training will be done separately on each atlas, '
                             'rather than combining the patches from different atlases to '
                             'avoid intensity scaling between them.')
    required.add_argument('--psize',required=True, type=int, nargs='+', dest='PATCHSIZE',
                        help='2D/3D patch size, e.g. --psize 32 32 32. \n**** Patch sizes must be multiple of 16.****')

    required.add_argument('--model', required=True, type=str, dest='MODEL',
                        help='Training model, options are Unet, DenseNet, Inception, RCAN, UNET++, EDSR, AttentionUnet')
    required.add_argument('--o', required=True, action='store', dest='OUTDIR',
                        help='Output directory where the trained models are written.')
    # Optional arguments
    optional = parser.add_argument_group('Optional arguments')

    optional.add_argument('--modalities', required=False, action='store', dest='MODAL', default=None,
                          help='A string of input and target image modalities. Accepted modalities are T1/T2/PD/FL/CT/UNK/MIC. '                               
                               'Default unk,unk,..., i.e. no normalization')
    optional.add_argument('--gpu', required=False, action='store', dest='GPU', type=str, default='0',
                        help='GPU id or ids to use for training. Example --gpu 1 indicates gpu with id 1 will be used '
                             'for training. For multi-gpu training, use comma separated list, such as --gpu 2,3,4.')
    optional.add_argument('--maxpatch', required=False, default=50000, type=int, dest='MAXPATCH', action='store',
                        help='Maximum number of patches to be collected from each atlas. Default is 50,000.')
    optional.add_argument('--basefilters', required=False, dest='BASEFILTER', default=16, type=int, action='store',
                        help='Base number of convolution filters to be used. Usually 8-32 works well. '
                             'Maximum number of filters in last conv block in Unet is 16 x BASEFILTERS. ')
    optional.add_argument('--batchsize', required=False, default=64, type=int, dest='BATCHSIZE', action='store',
                        help='Batch size. Default 64. Usually 50-150 works well.')
    optional.add_argument('--epoch', required=False, default=50, type=int, dest='EPOCH', action='store',
                        help='Maximum number of epochs to run. Default is 20. Usually 20-50 works well.')
    optional.add_argument('--loss', required=False, default='mae', type=str, dest='LOSS', action='store',
                        help='Loss type. For segmentation/labeling, use BCE (binary cross-entropy). For denoising, use '
                             'MSE or MAE. Default MAE.')
    optional.add_argument('--initmodel', required=False, dest='INITMODEL', default='None',
                        help='Pre-trained model to initilize, if available.')
    optional.add_argument('--lr', required=False, dest='LR', default=0.0001, type=float,
                        help='Learning rate. Default is 0.0001.')

    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)

    results = parser.parse_args()
    outdir = os.path.expanduser(results.OUTDIR)
    outdir = os.path.abspath(outdir)
    if os.path.isdir(outdir) == False:
        print("Output directory does not exist. I will create it.")
        os.makedirs(outdir)

    atlasdir = os.path.expanduser(os.path.realpath(results.ATLASDIR))


    psize = np.asarray(results.PATCHSIZE, dtype=int)
    if np.array_equal(psize, (psize // 16) * 16) == False:
        sys.exit('Error: Patch size must be multiple of 16.')

    #if np.array_equal(psize, (psize//16)*16) == False and results.MODEL.lower() in ['unet','fpn']:
    #    sys.exit('Error: Patch size must be multiple of 16.')

    # For biowulf, SLURM will set GPU id, so don't change it
    #print(os.getenv('CUDA_VISIBLE_DEVICES'))
    if os.getenv('CUDA_VISIBLE_DEVICES') is None or len(os.getenv('CUDA_VISIBLE_DEVICES')) == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = results.GPU
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(os.getenv('CUDA_VISIBLE_DEVICES')))
    else:
        print('SLURM already sets GPU id to {}, I will not change it.'.format(os.getenv('CUDA_VISIBLE_DEVICES')))
        results.GPU = os.getenv('CUDA_VISIBLE_DEVICES')

   
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    numgpu = len(str(results.GPU).split(','))


    #x = str(results.MODAL)
    #y = x.split(',')
    #l = len(y)
    #modal = []
    #for j in range(0, l):
    #    modal.append(str(y[j]).upper())
    if results.MODAL is not None:
        x = str(results.MODAL)
        y = x.split(',')
        l = len(y)
        modal = []
        for j in range(0, l):
            modal.append(str(y[j]).upper())
        nummodal = len(modal)
    else:
        #@TODO: Check the atlas directory for atlasXX_M1, atlasXX_M2 etc images and determine number of channels.
        nummodal = 1 # Hardcode a single channel for the time being
        modal = []
        for i in range(nummodal+1):
            modal.append('UNK')


    if len(modal)<2:
        sys.exit('ERROR: Number of modalities must be at least 2 (N+1, N=number of inputs, 1 for output).')
    results.BATCHSIZE = results.BATCHSIZE * numgpu
    if str(results.LOSS).lower() not in ['mse','mae','bce', 'ssim']:
        sys.exit('Error: Loss must be either one of MSE/MAE/BCE/SSIM.')

    if str(results.MODEL).lower() not in ['unet','densenet','inception', 'edsr','rcan','attentionunet','unet++']:
        sys.exit('Error: Model must be either Unet/AttentionUnet/Unet++/Inception/RCAN/EDST/Densenet.')

    print('Atlas Directory     =', atlasdir)
    print('Training model      =', results.MODEL.upper())
    print('Number of atlases   =', results.NUMATLAS)
    print('Output Directory    =', outdir)
    print('Number of GPUs      =', numgpu)
    print('GPU IDs             =', results.GPU)
    print('Patch Size          =', psize)
    print('Image Modalities    =', modal)
    print('Max number of patch =', results.MAXPATCH)
    print('Num of base filters =', results.BASEFILTER)
    print('Batch size          =', results.BATCHSIZE)
    print('Number of epochs    =', results.EPOCH)
    print('Loss                =', str(results.LOSS).lower())
    print('Learning rate       =', str(results.LR))
    print('Pre-trained model   =', str(results.INITMODEL))


    opt = {'numatlas': results.NUMATLAS,
           'outdir': outdir,
           'model': results.MODEL.lower(),
           'modalities': modal,
           'atlasdir': atlasdir,
           'numgpu': numgpu,
           'gpuid' : results.GPU,
           'patchsize': psize,
           'maxpatch': results.MAXPATCH,
           'basefilter': results.BASEFILTER,
           'batchsize': results.BATCHSIZE,
           'epoch': results.EPOCH,
           'loss': results.LOSS,
           'initmodel': results.INITMODEL,
           'lr': results.LR,
           }


    main(opt)