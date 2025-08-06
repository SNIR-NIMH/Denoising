import tensorflow as tf
import numpy as np
import os
import sys
import nibabel as nifti
import statsmodels.api as sm
from tensorflow import keras
import keras.backend as K
#from keras.utils import multi_gpu_model
from scipy.signal import argrelextrema
from tensorflow.keras import Input, Model
from sklearn.mixture import GaussianMixture
import h5py
from pytiff import Tiff
from skimage import io
from tqdm import tqdm
from glob import glob
from PIL import Image
from keras.backend import  set_session, get_session, clear_session
#import keras_contrib.backend as KC
from skimage import io
io.use_plugin('pil')


'''
class ModelMGPU(Model):

    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)
'''




def reset_keras():
    sess = get_session()
    clear_session()
    # sess.close()

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)


def write_image(img, imagename, niftiobj=None, compression=0, bigtifflag=True):
    _, ext = os.path.splitext(imagename)
    if ext == '.nii' or ext == '.gz':
        if niftiobj is not None:

            niftiobj.header['bitpix'] = 32
            niftiobj.header['datatype'] = 16
            temp = nifti.Nifti1Image(img, niftiobj.affine, niftiobj.header)
            nifti.save(temp, imagename)
        else:
            temp = nifti.Nifti1Image(img, affine=np.eye(4))
            nifti.save(temp, imagename)
    elif ext == '.h5':
        img = np.transpose(img, [2, 0, 1])  # H5 file have slices first, so that they can be opened in Fiji
        f = h5py.File(imagename, 'w')
        # f.create_dataset("data", data=img, compression='gzip', compression_opts = 1, dtype=np.float32)
        if str(img.dtype) == 'float32':
            f.create_dataset("data", data=img, dtype=np.float32)
        elif str(img.dtype) == 'uint16':
            f.create_dataset("data", data=img, dtype=np.uint16)
        else:
            img = np.asarray(img, dtype=np.float32)
            f.create_dataset("data", data=img, dtype=np.float32)
        f.close()
    elif ext == '.tif':
        img = np.transpose(img, [2, 0, 1])  # tif X-Y and python X-Y are flipped, also Z comes first
        if str(img.dtype) != 'uint16':
            #img = np.asarray(img, dtype=np.uint16)  # Tiff images are always uint16
            print('WARNING: Input image type is not UINT16, rather %s' %(str(img.dtype).upper()))
        if compression == 0:
            io.imsave(imagename, img, check_contrast=False, bigtiff=bigtifflag)
        else:
            io.imsave(imagename, img, check_contrast=False, bigtiff=bigtifflag, compression=compression)

    else:
        sys.exit('Image (%s) must be either nii, nii.gz, tif, or h5.' % (imagename))

#@TODO: Currently read_image only works for reading 3D images.
def read_image(imagename):
    _, ext = os.path.splitext(imagename)
    if ext == '.nii' or ext == '.gz':
        print('Reading {} with nibabel'.format(imagename))
        temp = nifti.load(imagename)
        img = temp.get_data().astype(dtype=np.float32)

    elif ext == '.h5':
        f = h5py.File(imagename, 'r')
        dataset = list(f.keys())
        print('Reading {} with h5py'.format(imagename))
        if len(dataset) > 1:
            print('WARNING: More than one key found in the h5 file {}'.format(dataset[0]))
            print('WARNING: Reading only the first key {} '.format(dataset[0]))
        img = f[dataset[0]]
        img = np.transpose(img, [1, 2, 0])  # for h5 images, the order is zxy

    elif ext == '.tif' or ext == '.tiff':
        try:
            #print('Reading {} with skimage'.format(imagename))
            img = io.imread(imagename,  is_ome=False, plugin='pil') # Why plugin pillow? The default is tifffile, which returns
            # LIBDEFLATE_INSUFFICIENT_SPACE  error if the image is deflate compressed.

            # If the number of slices is 3 or 4, imread assumes a RGB/RGBA image, so keeps X-Y-C orientation.
            # In all other cases, imread assumes grayscale multi-slice image, so keeps Z-X-Y orientation
            if len(img.shape)>2:  # Try this only for 3D images, for 2D images img.shape is 2x1 vector
                if img.shape[2] !=3 and img.shape[2] != 4:
                    img = np.transpose(img, [1, 2, 0])  # for 3D tif images, the order is zxy
            else:
                img = np.expand_dims(img,2) # why? skimage.io will create 2D image with 2D array, unlike pytiff where
                # I create 3D array. pad_image requires 3D array

        except Exception as e:
            print('Reading with skimage.io.imread failed.')
            print('%s' %(str(e)))
            print('Reading {} with pytiff/Tiff'.format(imagename))
            handle = Tiff(imagename, 'r')
            dim = (handle.size[0], handle.size[1], handle.number_of_pages) # number of slices is last
            img = np.zeros(dim, dtype=np.uint16)
            for j in range(0, dim[2]):
                handle.set_page(j)
                img[:, :, j] = handle[:]
            #img = np.squeeze(img) # DONT DO IT.


    elif os.path.isdir(imagename) == True:
        print('WARNING: Assuming input as a directory containing 2D tif images.')
        x = os.path.join(imagename, '*.tif')
        files = sorted(glob(x))
        N = len(files)
        # x = io.imread(files[-1])
        # Don't use io.imread to open the first tiff image,
        # it could be OME-TIFF, in which case io.imread will read the whole stack
        #@TODO: Use a recursive read_image or add the exceptions
        x = np.asarray(Image.open(files[0]), dtype=np.uint16)

        dim = x.shape
        dim = np.asarray((dim[0],dim[1],N), dtype=int)
        img = np.zeros(dim,dtype=np.uint16)

        for i in range(0,N):
            img[:,:,i] = np.asarray(Image.open(files[i]), dtype=np.uint16)
    else:
        img = None

    return img


def dice_coeff(y_true, y_pred):
    smooth = 0.0001
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


def dice_coeff_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)


def focal_loss(gamma):
    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-6
        alpha = 0.5
        y_pred = K.clip(y_pred, eps,
                        1. - eps)  # improve the stability of the focal loss and see issues 1 for more information
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0), axis=-1)

    return focal_loss_fixed


def focal_loss_fixed(y_true, y_pred):
    gamma = 2
    eps = 1e-6
    alpha = 0.5
    y_pred = K.clip(y_pred, eps,
                    1. - eps)  # improve the stability of the focal loss and see issues 1 for more information
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0), axis=-1)


def normalize_image(vol, contrast):
    # All MR images must be non-negative. Sometimes cubic interpolation may introduce negative numbers.
    # This will also affect if the image is CT, which not considered here. Non-negativity is required
    # while getting patches, where nonzero voxels are considered to collect patches.

    if contrast.lower() not in ['t1', 't1c', 't2', 'pd', 'fl', 'flc', 'mic']:
        print("Contrast must be either T1,T1C,T2,PD,FL,FLC,or MIC. You entered %s. Returning original volume." % contrast)
        return vol, 1.0

    if contrast.lower() == 'mic':  # microscopy images are usually too large, so sample 100k values, also assume they are non-zero
        dim = vol.shape
        S = (500000 // dim[2]) * dim[2]
        s = S // dim[2]
        temp = np.zeros((S,), dtype=np.float32)
        for k in range(0, dim[2]):
            temp[s * k:s * k + s] = np.random.choice(np.ndarray.flatten(vol[:, :, k]), s)
        temp = temp[temp > 0]
        temp = np.asarray(temp, dtype=np.float32)
    else:
        vol[vol < 0] = 0
        temp = vol[np.nonzero(vol)].astype(np.float32)

    q2 = np.percentile(temp, 99)
    q1 = np.percentile(temp, 1)
    temp = temp[temp <= q2]
    temp = temp[temp >= q1]
    temp = temp.reshape(-1, 1)
    bw = (q2 - q1) / 80
    q = q2
    print("99th quantile is %.2f, 1st quantile is %.2f, gridsize = %.4f" % (q2, q1, bw))

    kde = sm.nonparametric.KDEUnivariate(temp)

    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    x_mat = 100.0 * kde.density
    y_mat = kde.support

    indx = argrelextrema(x_mat, np.greater)
    indx = np.asarray(indx, dtype=int)
    heights = x_mat[indx][0]
    peaks = y_mat[indx][0]
    peak = 1.00
    print("%d peaks found." % (len(peaks)))

    if contrast.lower() in ["t1", "t1c"]:
        print("Double checking peaks with a GMM.")
        gmm = GaussianMixture(n_components=3, covariance_type='spherical', tol=0.001,
                              reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', precisions_init=None,
                              weights_init=(0.33, 0.33, 0.34),
                              means_init=np.reshape((0.2 * q, 0.5 * q, 0.95 * q), (3, 1)),
                              warm_start=False, verbose=0, verbose_interval=1)
        gmm.fit(temp.reshape(-1, 1))
        m = gmm.means_[-1]
        peak = peaks[-1]
        if m / peak < 0.75 or m / peak > 1.25:
            print("WARNING: WM peak could be incorrect (%.4f vs %.4f). Please check." % (m, peak))
            peaks = m
        peak = peaks[-1]
        print("Peak found at %.4f for %s" % (peak, contrast))
    elif contrast.lower() in ['t2', 'pd', 'fl', 'flc']:
        peak_height = np.amax(heights)
        idx = np.where(heights == peak_height)
        peak = peaks[idx]
        print("Peak found at %.4f for %s" % (peak, contrast))
    elif contrast.lower() == 'mic':
        peak = peaks[-1]
        print("Peak found at %.4f for %s" % (peak, contrast))
    else:
        print("Contrast must be either T1,T1C,T2,PD,FL, or FLC. You entered %s. Returning original volume." % contrast)
    return vol / peak, peak


def Crop(vol, bg=0, padsize=0):
    dim = vol.shape
    cpparams = []
    s = 0
    for k in range(0, dim[0]):
        if np.sum(vol[k, :, :]) == bg:
            s = s + 1
        else:
            break
    cpparams.append(np.maximum(s - padsize, 0))
    s = dim[0] - 1
    for k in range(dim[0] - 1, -1, -1):
        if np.sum(vol[k, :, :]) == bg:
            s = s - 1
        else:
            break
    cpparams.append(np.minimum(s + padsize, dim[0] - 1))
    s = 0
    for k in range(0, dim[1]):
        if np.sum(vol[:, k, :]) == bg:
            s = s + 1
        else:
            break
    cpparams.append(np.maximum(s - padsize, 0))
    s = dim[1] - 1
    for k in range(dim[1] - 1, -1, -1):
        if np.sum(vol[:, k, :]) == bg:
            s = s - 1
        else:
            break
    cpparams.append(np.minimum(s + padsize, dim[1] - 1))
    s = 0
    for k in range(0, dim[2]):
        if np.sum(vol[:, :, k]) == bg:
            s = s + 1
        else:
            break
    cpparams.append(np.maximum(s - padsize, 0))
    s = dim[2] - 1
    for k in range(dim[2] - 1, -1, -1):
        if np.sum(vol[:, :, k]) == bg:
            s = s - 1
        else:
            break
    cpparams.append(np.minimum(s + padsize, dim[2] - 1))
    vol2 = vol[cpparams[0]:cpparams[1], cpparams[2]:cpparams[3], cpparams[4]:cpparams[5]]
    return vol2, cpparams, dim


# Padding 3D images in 3D, the dim says if all 3 dimensions need to be padded or not
def pad_image(vol, padsize=0, dim=3):
    '''
    :param vol: Must be a 3D volume
    :param padsize: a scalar int
    :param dim: Either 2 or 3, indicating if the padding to be done in 3D or just x-y dimensions. This is useful
    for padding anisotropic image with 2D patch size
    :return:
    '''

    if dim == 3:
        if len(padsize) == 1: # If padsize is a single number, than means isotropic padding
            padsize = (padsize, padsize, padsize)
    else:
        padsize = (padsize, padsize, 0,)
    padsize = np.asarray(padsize, dtype=int)
    origdim = np.asarray(vol.shape, dtype=int)
    dim2 = origdim + 2 * padsize

    temp = np.zeros(dim2, dtype=np.float32)
    temp[padsize[0]:origdim[0] + padsize[0], padsize[1]:origdim[1] + padsize[1], padsize[2]:origdim[2] + padsize[2]] = vol

    return temp

'''
# Not to confuse with padding 2D images. They can be combined, but @TODO
def pad_image2D(img, padsize=0):    

    padsize = (padsize, padsize)
    padsize = np.asarray(padsize, dtype=int)
    origdim = np.asarray(img.shape, dtype=int)
    dim2 = origdim + 2 * padsize
    temp = np.zeros(dim2, dtype=np.float32)
    temp[padsize[0]:origdim[0] + padsize[0], padsize[1]:origdim[1] + padsize[1]] = img
    return temp
'''

def ssim_metric(y_true, y_pred):
    k1 = 0.0001
    k2 = 0.0009
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    u_true = K.mean(y_true)
    u_pred = K.mean(y_pred)
    var_true = K.var(y_true)
    var_pred = K.var(y_pred)
    covar_true_pred = K.mean(y_true * y_pred) - u_true * u_pred
    ssim = (2 * u_true * u_pred + k1) * (2 * covar_true_pred + k2)

    denom = ((K.square(u_true) + K.square(u_pred) + k1) * (var_pred + var_true + k2))
    ssim = ssim / denom  # no need for clipping, c1 and c2 make the denom non-zero
    return  ssim


def ssim_loss(y_true, y_pred):
    return 1 - ssim_metric(y_true, y_pred)
