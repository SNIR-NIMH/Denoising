import nibabel as nifti
import glob
import os
import sys
import random
import tempfile
import copy
from tqdm import tqdm
import numpy as np
from scipy.signal import argrelextrema
import argparse
from PIL import Image
import cv2
import sys
from pytiff import Tiff
import gc
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from keras import backend as K
from keras.layers import Dense, Input, Lambda
from keras.models import load_model, Model
import statsmodels.api as sm
import time, datetime
from skimage import io
from skimage.io import imread,imsave
from sklearn.mixture import GaussianMixture
import h5py
from pynvml import *
Image.MAX_IMAGE_PIXELS = 75000*75000


path = os.path.dirname(sys.argv[0])
path = os.path.abspath(path)
path = os.path.join(path, 'CNNUtils')
print('Appending {}'.format(path))
sys.path.append(path)
from CNNUtils.utils import reset_keras
from CNNUtils.utils import normalize_image, pad_image, Crop, read_image, write_image
from CNNUtils.utils import dice_coeff, dice_coeff_loss, focal_loss, focal_loss_fixed, ssim_loss, ssim_metric



def split_array(input, psize, chunksize):  # Input is a 4D array now, HxWxDxC
    adim = np.asarray(input.shape, dtype=int)
    numsplit = np.prod(chunksize)
    sdim = (numsplit, adim[0]//chunksize[0] + 2*psize[0]+1, adim[1]//chunksize[1] + 2*psize[1]+1, adim[2], adim[3])
    #print(adim)
    #print(sdim)
    #sys.exit()
    splitinput = np.zeros(sdim, dtype=np.float32)
    count = 0
    for i in range(0,chunksize[0]):
        for j in range(0,chunksize[1]):
            I1 = i*(adim[0]//chunksize[0]) - psize[0]
            I2 = (i+1)*(adim[0]//chunksize[0]) + psize[0]+1
            J1 = j*(adim[1]//chunksize[1]) - psize[1]
            J2 = (j+1)*(adim[1]//chunksize[1]) + psize[1] + 1
            delta = [0,0]
            if I1<0:
                delta[0] = -I1
                I1=0
            if I2>adim[0]:
                I2 = adim[0]
            if J1<0:
                delta[1] = -J1
                J1=0
            if J2>adim[1]:
                J2=adim[1]

            x = input[I1:I2, J1:J2, :, :]

            splitinput[count, delta[0]:x.shape[0]+delta[0], delta[1]:x.shape[1]+delta[1], :, : ] = x
            count = count + 1

    return splitinput


def ApplyModel3D(in_vol, model, network): # in_vol is HxWxDx1 matrix, this function will pad whatever it gets and runs the model

    dim = in_vol.shape
    #print(dim)
    #if str(network).lower() == 'unet' or str(network).lower() == 'attentionunet' or str(network).lower() == 'unet++' :
    if str(network).lower() == 'unet' or str(network).lower() == 'attentionunet' \
            or str(network).lower() == 'unet++' or str(network).lower() == 'inception' : # The inception architecture is unet-like
        adim = np.ceil(np.asarray(dim[0:3], dtype=float) / 16) * 16
        # All Unets need dimensions multiple of 16
    else:
        adim = copy.deepcopy(dim)
    batch_dim = np.asarray((1, adim[0], adim[1], adim[2], dim[3]), dtype=int)
    batch = np.zeros(batch_dim, dtype=np.float32)


    batch[0, 0:dim[0], 0:dim[1], 0:dim[2], 0:dim[3]] = in_vol
    #print(batch.shape)
    pred = model.predict_on_batch(batch)
    out_vol = pred[0, 0:dim[0], 0:dim[1], 0:dim[2], 0]
    #print(out_vol.shape)


    '''
    psize = np.asarray(psize, dtype=int)
    dsize = psize // 2
    dsize = np.asarray(dsize, dtype=int)
    dim = in_vol.shape
    #print('Input shape = {}'.format(dim))
    out_vol = np.zeros((dim[0], dim[1], dim[2]), dtype=np.float32)

    batch_dim = (1, in_vol.shape[0], in_vol.shape[1], psize[2], in_vol.shape[3])
    batch = np.zeros(batch_dim, dtype=np.float32)

    for k in tqdm(list(range(dsize[2], in_vol.shape[2] - dsize[2] - 1))):
        batch[0, :, :, :, :] = in_vol[:, :, k - dsize[2]:k + dsize[2], :]
        # if np.ndarray.sum(batch) > 0:
        pred = model.predict_on_batch(batch)
        out_vol[:, :, k] = pred[0, :, :, dsize[2], 0]

    
    '''
    return out_vol


def ApplyModel2D(vol, model, progressbar=True):
    dim = vol.shape
    #print('Input shape = {}'.format(dim))
    dim = np.asarray(dim, dtype=int)
    dim2 = (dim[0], dim[1], dim[2])
    outvol = np.zeros(dim2, dtype=np.float32)

    invol = np.zeros((1, dim[0], dim[1], dim[3]), dtype=np.float32)
    #print(invol.shape)
    if progressbar:
        for k in tqdm(range(0, dim[2])):
            for t in range(0, dim[3]):
                invol[0, :, :, t] = vol[:, :, k, t]
            pred = model.predict(invol, verbose=None)
            if k==0 and t==0: # print only once
                mem = nvmlDeviceGetMemoryInfo(gpuhandle)
                #print('Used = {} MB, Free = {} MB'.format(int(mem.used / (1024 ** 2)), int(mem.free / (1024 ** 2))))
            outvol[:, :, k] = pred[0, :, :, 0]
    else:
        for k in range(0, dim[2]):
            for t in range(0, dim[3]):
                invol[0, :, :, t] = vol[:, :, k, t]
            pred = model.predict(invol, verbose=None)
            if k==0 and t==0: # print only once
                mem = nvmlDeviceGetMemoryInfo(gpuhandle)
                #print('Used = {} MB, Free = {} MB'.format(int(mem.used / (1024 ** 2)), int(mem.free / (1024 ** 2))))
            outvol[:, :, k] = pred[0, :, :, 0]
    return outvol


parser = argparse.ArgumentParser(description='Model Prediction')

# Required inputs
parser.add_argument('--im','-i', required=True, dest='IMAGES', nargs='+', type=str,
                    help='Input images, nifti (.nii or .nii.gz) or TIF (.tif or .tiff). The order must'
                         'be same as the order of atlasXX_M1.nii.gz, atlasXX_M2.nii.gz images, i.e. 1st'
                         'input must be of channel M1, second M2, etc. For microscopy images (--modalities mic),'
                         'a single tif image directory containing multiple 2D tif image slices is also acceptable. ')

parser.add_argument('--o','-o', required=True, action='store', dest='OUTPUT',
                    help='Output filename, e.g. somefile.nii.gz or somefile.tif where the result will be written. '
                         'If the image is large (e.g. stitched images), use a folder as output, e.g. /home/user/output_folder/ '
                         'where 2D slices will be written. Output can be NIFTI only if the input is also NIFTI.')
parser.add_argument('--model', required=True,  dest='MODEL', type=str,
                    help='Trained model (.h5) files. Only a single model is accepted')
parser.add_argument('--network', required=True, dest='NETWORK', type=str,
                    help = 'Type of network used for training. Options are Unet, DenseNet, Inception, RCAN, UNET++, EDSR, AttentionUnet')

parser.add_argument('--psize', required=True, type=int, nargs='+', dest='PATCHSIZE',
                    help='2D/3D patch size used for training, e.g. 16 16 16, separated by space. '
                         'If the training model was U-net, the patch size must be multiple of 16, otherwise '
                         'there will be size mismatch error. Patch size must be even.')
# Optional inputs
parser.add_argument('--modalities', required=False, action='store', dest='MODAL', default=None,
                    help='(Optional) A comma separated string of input image modalities. '
                         'Accepted modalities are T1/T2/PD/FL/CT/MIC/UNK. Example is t1,t2. This is the same '
                         'as entered during training (without the output modality). This is needed for proper '
                         'normalization of images. Default: If you don''t want images to be normalized (same for training), '
                         'then use UNK (unknown) as modality.')
parser.add_argument('--gpu', required=False, action='store', dest='GPU', type=int, default=0,
                    help='(Optional) GPU id to use. Default is 0.')
parser.add_argument('--chunks', required = False, dest='CHUNKS', type=int, nargs='+', default=[0,0],
                    help='(Optional) If the input image size is too large (such as stitched images) to fit into GPU memory, '
                         'it can be chunked using "--chunks nh nw" argument. E.g. --chunks 3 2 will split a '
                         'HxWxD image into overlapping (H/3)x(W/2)xD chunks, apply the trained models on '
                         'each chunk serially, then join the chunks. This option works only if (1) the input and '
                         'outout images are both TIF (either 3D or a folder), (2) only one channel is available. '
                         '**Normally, if image is not chunked, total memory required is 6 times the size of the image.** '
                         'Default: no chunking')
parser.add_argument('--float', required = False, dest='FLOAT', action='store_true',
                    help='(Optional) Use --float to save output images as FLOAT32. Default is UINT16. This is useful '
                         'if the dynamic range of the training data is small. Note, saving as FLOAT32 images will '
                         'approximately double the size of the image.')
parser.add_argument('--compress', required=False, dest='COMPRESS', action='store_true',
                    help='(Optional) If --compress is used, the output Tif images will be compressed. ')


if len(sys.argv) < 2:
    parser.print_usage()
    sys.exit(1)

results = parser.parse_args()
t1 = time.process_time()

if os.getenv('CUDA_VISIBLE_DEVICES') is None or len(os.getenv('CUDA_VISIBLE_DEVICES')) == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(results.GPU)
    print('Setting CUDA_VISIBLE_DEVICES to {}'.format(os.getenv('CUDA_VISIBLE_DEVICES')))
else:
    print('SLURM already sets GPU id to {}, I will not change it.'.format(os.getenv('CUDA_VISIBLE_DEVICES')))
    results.GPU = int(os.getenv('CUDA_VISIBLE_DEVICES').split(',')[0])

if results.COMPRESS == True:
    results.COMPRESS = 'zlib'
else:
    results.COMPRESS = 0


reset_keras()
nvmlInit()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
gpuhandle = nvmlDeviceGetHandleByIndex(results.GPU)


psize = np.asarray(results.PATCHSIZE, dtype=int)

if len(psize) != 2 and len(psize) != 3:
    print('ERROR: Patch size must be a tuple of triplet. You entered a {}-dim array.'.format(len(psize)))
    sys.exit()

for i in range(0, len(psize)):
    psize[i] = (psize[i] // 2) * 2


results.MODEL = os.path.abspath(os.path.expanduser(results.MODEL))

print(("Trained model found at {}".format(results.MODEL)))

results.IMAGES = [os.path.abspath(os.path.expanduser(im)) for im in results.IMAGES]
print(("%d images found at" % (len(results.IMAGES))))
for im in results.IMAGES:
    print(im)

if results.MODAL is not None:
    x = str(results.MODAL)
    y = x.split(',')
    l = len(y)
    modal = []
    for j in range(0, l):
        modal.append(str(y[j]).upper())
    nummodal = len(modal)
else:
    modal=[]
    for i in range(len(results.IMAGES)):
        modal.append('UNK')
    nummodal = len(results.IMAGES)

if len(results.IMAGES) != nummodal:
    print(('ERROR: Number of images (%d) must be same as the number of modalities (%d).' % (
        len(results.IMAGES), nummodal)))
    sys.exit()



# Checking on input images.
if os.path.isfile(results.IMAGES[0]):
    _, ext1 = os.path.splitext(os.path.basename(results.IMAGES[0]))
    if ext1 not in ['.nii','.gz','.tif']:
        print('ERROR: Input image type {} not supported'.format(ext1) )
        sys.exit('ERROR: Input must be a NIFTI (e.g. somefile.nii or somefile.nii.gz) or a TIF file (somefile.tif)')

if os.path.isfile(results.IMAGES[0]) == True:
    _, ext = os.path.splitext(os.path.basename(results.IMAGES[0]))
    if ext =='.nii' or ext == '.gz':  # assuming nifti images are small
        outvol = read_image(results.IMAGES[0])
        origdim = outvol.shape
    else:
        handle = Tiff(results.IMAGES[0], 'r')
        origdim = (handle.size[0], handle.size[1], handle.number_of_pages)  # number of slices is last
        print(origdim)

    input_is_file = True
elif os.path.isdir(results.IMAGES[0]) == True:
    s = os.path.join(results.IMAGES[0], '*.tif')
    files=sorted(glob.glob(s))
    if len(files) ==0 : # .tif files not found
        s = os.path.join(results.IMAGES[0], '*.tiff')
        files=sorted(glob.glob(s))

    if len(files) ==0:
        print('ERROR: .tif or .tiff files not found in {}'.format(results.IMAGES[0]))
        sys.exit()
    try:
        print('Reading image with PIL.Image')
        x = Image.open(files[-1])
    except:
        try:
            print('Reading image with skimage.io.imread')
            print(files[-1])
            x = io.imread(files[-1])
        except:
            sys.exit('ERROR: Image can not be read with PIL.Image or skimage.io.imread')

    x = np.asarray(x,dtype=np.uint16)
    origdim = x.shape
    origdim = (origdim[0],origdim[1],len(files))
    input_is_file = False
else:
    sys.exit('ERROR: Can not open file {}'.format(results.IMAGES[0]))


# Checking on output image.
outname = os.path.realpath(os.path.abspath(os.path.expanduser(results.OUTPUT)))
if os.path.isfile(outname) == True:
    print('ERROR: File exists: %s' % (outname))
    sys.exit('ERROR: I will not overwrite.')
_, ext2 = os.path.splitext(os.path.basename(outname))
if ext2 not in ['.nii','.gz','.tif','.tiff']:
    print('WARNING: Output is not a NIFTI (e.g. somefile.nii or somefile.nii.gz) or a TIF file (somefile.tif)')
    print('WARNING: Assuming output is a folder, where 2D tif slices will be written.')
    os.makedirs(outname, exist_ok=True)
    output_is_file = False
else:
    output_is_file = True


print('Output will be written at {}'.format(outname))
if results.FLOAT == True:
    print('Output will be saved as 32-bit FLOAT images.')
else:
    print('Output will be saved as 16-bit UINT images.')



if results.CHUNKS[0] > 0 and results.CHUNKS[1]>0:
    chunksize = np.asarray(results.CHUNKS, dtype=int)
    print('Image will be split into {} x {} chunks.'.format(results.CHUNKS[0],results.CHUNKS[1]))
    numsplit = np.prod(chunksize)

    if os.path.isfile(results.IMAGES[0]):
        _, ext = os.path.splitext(os.path.basename(results.IMAGES[0]))
        if ext != '.tif' and ext != '.tiff':
            sys.exit('ERROR: Image chunking only works for 3D TIF images or a folder containing 2D tifs.')
    # @TODO : Ideally big nifti images are never used. Chunking can work on any set of 2D slices.
    if nummodal > 1:
        sys.exit('ERROR: Image chunking only works for a single channel input.')

else:
    numsplit = 0



origdim = np.asarray(origdim, dtype=int)
padsize = psize // 2
dim = copy.deepcopy(origdim)

dict = {"tf": tf,
        "dice_coeff": dice_coeff,
        "dice_coeff_loss": dice_coeff_loss,
        "focal_loss": focal_loss,
        "focal_loss_fixed": focal_loss_fixed,
        "ssim_loss": ssim_loss,
        "ssim_metric": ssim_metric,

        }

if numsplit == 0:   # Assume images are small, and will fit into memory, probably does not work for large images

    # If the folder contains different size images, chunking must be enabled

    dim = (dim[0], dim[1], dim[2], nummodal)
    dim = np.asarray(dim, dtype=int)

    if len(psize) == 3:
        # dim = ((origdim)//16 + 1)*16 + 2*padsize   # Padded dimension
        dim[0:2] = np.ceil(np.asarray(origdim[0:2], dtype=float) / 16) * 16 + 2 * padsize[0:2]
        dim[2] = origdim[2] + psize[2]  # Don't add psize[2] padding, add dsize[2] padding on both sides
    else:
        # dim[0:2] = ((origdim[0:2])//16 + 1)*16 + 2*padsize   # Padded dimension
        dim[0:2] = np.ceil(np.asarray(origdim[0:2], dtype=float) / 16) * 16 + 2 * padsize
    dim = np.asarray(dim, dtype=int)
    print('WARNING: Chunking is not mentioned. I will assume input image is small enough to fit into memory.')
    gb=np.ceil(np.prod(dim) * 4.0 * 3 // (1024 ** 3))
    print('WARNING: Approximate memory required = %d GB' % (gb))
    if gb>25:
        print('********************************************************************************')
        print('')
        print('**** WARNING: More than 25GB memory is required. Are you sure? Try chunking instead. ****')
        print('**** Chunking produces memory efficient result with significantly less memory        ****')
        print('')
        print('********************************************************************************')


    vol4d = np.zeros(dim, dtype=np.float32)  # It is a 4D array HxWxDxC
    print("Original dimension = {}".format(origdim))
    print("Padded dimension   = {}".format(dim))  # Pad original image if there is no chunking
    #@TODO: Only padding x-y dimensions for the time being, this works best for 2D
    outvol = np.zeros(origdim, dtype=np.float32)
    peaks = np.zeros(len(results.IMAGES),dtype=np.float32)


    for t in range(0, nummodal):

        print('Reading image..')
        temp = read_image(results.IMAGES[t])
        temp = np.asarray(temp, dtype=np.float32)
        if np.min(temp) < 0:
            print('ERROR: Minimum value of image is {}'.format(np.min(temp)))
            sys.exit('ERROR: Images must be non-zero. {}'.format(results.IMAGES[t]))
        print('Normalizing image..')
        temp, peaks[t] = normalize_image(temp, modal[t])
        dsize = psize//2
        if len(psize) ==3:
            vol4d[padsize[0]:origdim[0] + padsize[0], padsize[1]:origdim[1] + padsize[1], dsize[2]:origdim[2]+dsize[2], t] = temp
        else:
            vol4d[padsize[0]:origdim[0] + padsize[0], padsize[1]:origdim[1] + padsize[1], 0:origdim[2], t] = temp


        temp = 0  # free some memory, especially useful for large microscopy images
    gc.collect()
    print(('Padded image size = %s' % (str(vol4d.shape))))

else:     # Images are large (e.g. stitched images), so either use pytiff or read slices. Also there is a single channel

    # For large images, use some fancy normalization technique on downsampled images
    peaks = [1]
    if (modal[0]).upper() != 'UNK':
        print('Reading part of the image to compute normalization factor.')
        n = np.ceil(origdim[0]/1000)  # Getting approximately a 1000x1000x100 image to normalize
        m = np.ceil(origdim[2]/100)
        temp = np.zeros((origdim[0]//n, origdim[1]//n, origdim[2]//m ), dtype=np.uint16)
        count = 0
        for j in tqdm(range(0, origdim[2], m)):

            if input_is_file == True:
                handle.set_page(j)
                x = handle[:]
            else:
                x = np.asarray(read_image(files[j]), dtype=np.uint16)
            x = cv2.resize(x,(origdim[1]//n, origdim[0]//n),cv2.INTER_NEAREST)
            temp[:, :, count] = x
            count = count + 1
        print('Computing scaling factor based on a {} size image.'.format(str(temp.shape)))
        _, peaks[0] = normalize_image(temp, modal[0])


    else:
        peaks[0] = 1.0

    dim = (dim[0], dim[1], dim[2], nummodal)
    dim = np.asarray(dim, dtype=int)

    if len(psize) == 3:
        # dim = ((origdim)//16 + 1)*16 + 2*padsize   # Padded dimension
        dim[0:2] = np.ceil(np.asarray(origdim[0:2], dtype=float) / 16) * 16 + 2 * padsize[0:2]
        # adding 2*psize is not necessary, usually it should be psize+1, but psize must be even, so simply pad by psize,
        # and read psize/2 slices
        #dim[2] = origdim[2] + 2 * psize[2]
        dim[2] = origdim[2] + psize[2]
    else:
        # dim[0:2] = ((origdim[0:2])//16 + 1)*16 + 2*padsize   # Padded dimension
        dim[0:2] = np.ceil(np.asarray(origdim[0:2], dtype=float) / 16) * 16 + 2 * padsize
    dim = np.asarray(dim, dtype=int)

    # No need to define vol4D here because it will be redefined later for every iteration, L545
    #vol4d = np.zeros(dim, dtype=np.float32)  # It is a 4D array HxWxDxC
    print("Original dimension = {}".format(origdim))
    print("Padded dimension   = {}".format(dim))  # Pad original image if there is no chunking


if numsplit == 0:
    # If the folder contains different size images, chunking must be enabled

    model = load_model(results.MODEL, custom_objects=dict)
    print('Predicting with model {}'.format(os.path.basename(results.MODEL)))


    if len(psize) == 2:
        syn = ApplyModel2D(vol4d, model, progressbar = True)
        outvol = outvol + syn[padsize[0]:padsize[0] + origdim[0], padsize[1]:padsize[1] + origdim[1], 0:origdim[2]]
        syn = 0

    else:
        #print(vol4d.shape)
        # @TODO This could be erroneous
        dsize=psize//2
        for k in tqdm(range(dsize[2],dim[2]-dsize[2])):
            #print(vol4d[:,:,k-dsize[2]:k+dsize[2]].shape)
            syn = ApplyModel3D(vol4d[:,:,k-dsize[2]:k+dsize[2]], model, results.NETWORK)
            syn = syn[:,:, dsize[2]]
            outvol[:,:,k-dsize[2]] = syn[padsize[0]:padsize[0] + origdim[0], padsize[1]:padsize[1] + origdim[1]]

        syn = 0

    model = None
    reset_keras()  # This is needed because load_model is called multiple times in a for loop
    del syn
    gc.collect()



else:

    #print('Splitting input array {} into {} x {} chunks.'.format(origdim, chunksize[0], chunksize[1]))
    #print('Splitting input array into {} x {} chunks.'.format(chunksize[0], chunksize[1]))
    #@TODO: To account for 2D images with different dimensions, origdim must be redefined every time only for folders
    #@TODO: with 2D images
    # If the folder contains different size images, chunking must be enabled
    # If the folder contains different size images, 3D output is not acceptable
    if len(psize) == 2:

        if output_is_file == True :
            outvol = np.zeros(origdim, dtype=np.float32)

        else:
            outputfilenames = [None]*origdim[2]
            for i in range(0, origdim[2]):
                if input_is_file:
                    s = os.path.basename(results.IMAGES[0])
                    s,_ = os.path.splitext(s)
                    s = s + '_Z' + str(i).zfill(4) + '.tif'
                    s = os.path.join(outname,s)
                    outputfilenames[i] = s
                else:
                    s = os.path.basename(files[i])
                    outputfilenames[i] = os.path.join(outname, s)


        model = load_model(results.MODEL, custom_objects=dict)


        for k in tqdm(range(0, origdim[2])):
            # To account for files with different dimensions in a folder, reread the file xy dims
            # This is slightly unoptimized because files are read twice
            if input_is_file == False:
                try:
                    # print('Reading image with PIL.Image')
                    x = Image.open(files[k])
                except:
                    x = io.imread(files[k], is_ome=False)
                x = np.asarray(x, dtype=np.uint16)
                newxydim = x.shape
            else:
                newxydim = copy.deepcopy((origdim[0],origdim[1]))
            #print(newxydim)
            #vol4d = np.zeros((origdim[0], origdim[1], 1, 1), dtype=np.float32)  # Number of channel=1, depth = 1 because 2D
            vol4d = np.zeros((newxydim[0], newxydim[1], 1, 1), dtype=np.float32)  # Number of channel=1, depth = 1 because 2D

            if input_is_file :
                handle.set_page(k)
                vol4d[:,:,0,0] = np.asarray(handle[:], dtype=np.float32)/peaks[0]
            else:
                try:
                    x = Image.open(files[k])
                except:
                    try:
                        x = io.imread(files[k], is_ome=False)
                    except:
                        sys.exit('ERROR: Image can not be read with PIL.Image or skimage.io.imread')

                vol4d[:,:,0,0] = np.asarray(x, dtype=np.float32)/peaks[0]
                #vol4d[:,:,0,0] = np.asarray(Image.open(files[k]), dtype=np.float32)/peaks[0]

            if output_is_file == False:
                #outvol = np.zeros((origdim[0], origdim[1]), dtype=np.float32)
                outvol = np.zeros((newxydim[0], newxydim[1]), dtype=np.float32)

            vol4d = split_array(vol4d, psize, chunksize)  # vol4d is a 4D array HxWxDxC, dim2
            # print('Chunked image size = {}'.format(vol4d.shape))  # vol4d becomes 5D array
            adim = np.asarray(vol4d.shape, dtype=int)


            adim2 = copy.deepcopy(adim[1:5])
            adim2[0:2] = np.ceil(adim2[0:2] / 16) * 16  # Chunked images are already overlapped by psize, so no need to add padsize as psize again
            #print('Padded chunked image size = {}'.format(adim2))  # vol4d becomes 5D array

            count=0
            for p in range(0, chunksize[0]):
                for q in range(0, chunksize[1]):
                    #print('Chunk {} of {}:'.format(count+1,numsplit))
                    #I1 = p * (origdim[0] // chunksize[0] )
                    #I2 = (p + 1) * (origdim[0] // chunksize[0])
                    #J1 = q * (origdim[1] // chunksize[1])
                    #J2 = (q + 1) * (origdim[1] // chunksize[1])
                    I1 = p * (newxydim[0] // chunksize[0])
                    I2 = (p + 1) * (newxydim[0] // chunksize[0])
                    J1 = q * (newxydim[1] // chunksize[1])
                    J2 = (q + 1) * (newxydim[1] // chunksize[1])
                    x = np.zeros(adim2, dtype=np.float32)
                    x[0:adim[1], 0:adim[2], :, :] = vol4d[count, :, :, :, :]


                    syn = ApplyModel2D(x, model, progressbar=False)
                    syn = syn[0:adim[1], 0:adim[2], :]
                    #syn = ApplyModel2D(vol4d[count, :, :, :, :], model, progressbar=False)
                    #print(syn.shape)


                    if output_is_file:
                        outvol[I1:I2, J1:J2, k] = syn[psize[0]:-psize[0] - 1, psize[1]:-psize[1] - 1, 0]
                    else:
                        outvol[I1:I2, J1:J2] = syn[psize[0]:-psize[0] - 1, psize[1]:-psize[1] - 1, 0]
                    syn = 0

                    if count==0 and k==0: # Print only once
                        mem = nvmlDeviceGetMemoryInfo(gpuhandle)
                        print('Used = {} MB, Free = {} MB'.format(int(mem.used / (1024 ** 2)),
                                                                  int(mem.free / (1024 ** 2))))

                    count = count + 1

            if output_is_file == False:

                #outvol = peaks[0] * outvol/ (len(results.MODEL))
                outvol = peaks[0] * outvol
                if results.FLOAT == False:
                    outvol[outvol > 65535] = 65535
                    outvol = np.asarray(outvol, dtype=np.uint16)

                io.imsave(outputfilenames[k], outvol, check_contrast=False, bigtiff=False, compression=results.COMPRESS)

    elif len(psize) ==3:  # full 3D model, load 2r+1 slices, apply, then use the r-th slice for pxpxr patches

        if output_is_file == True:
            outvol = np.zeros(origdim, dtype=np.float32)

        else:
            outvol = np.zeros((origdim[0], origdim[1]), dtype=np.float32)
            outputfilenames = [None] * origdim[2]
            for i in range(0, origdim[2]):
                if input_is_file:
                    s = os.path.basename(results.IMAGES[0])
                    s, _ = os.path.splitext(s)
                    s = s + '_Z' + str(i).zfill(4) + '.tif'
                    s = os.path.join(outname, s)
                    outputfilenames[i] = s
                else:
                    s = os.path.basename(files[i])
                    outputfilenames[i] = os.path.join(outname, s)

        model = load_model(results.MODEL, custom_objects=dict)
        #vol4d = np.zeros((origdim[0], origdim[1], 2*psize[2]+1, 1),  dtype=np.float32)  # Number of channel=1, depth = 1 because 2D
        vol4d = np.zeros((origdim[0], origdim[1], psize[2], 1), dtype=np.float32) # Try the z dimension as psize[2] instead
                                                                                  # of padding by psize[2]
        dsize = psize//2  # psize is always multiple of 16, so this is alright
        print('Prefetching..')
        '''
        if input_is_file:
            for j in range(0,psize[2]+1):
                handle.set_page(j)
                vol4d[:, :, j+psize[2], 0] = np.asarray(handle[:], dtype=np.float32) / peaks[0]
        else:
            for j in range(0, psize[2] + 1):
                x = io.imread(files[j], is_ome=False)
                vol4d[:, :, j+psize[2], 0] = np.asarray(x, dtype=np.float32) / peaks[0]
        # Padding by 1st slice
        for j in range(0,psize[2]):
            vol4d[:,:,j,0] = vol4d[:,:, psize[2],0]
        '''
        # Don't pad by psize[2], pad by dsize[2], see L708
        if input_is_file:
            for j in range(0,dsize[2]):
                handle.set_page(j)
                vol4d[:, :, j+dsize[2], 0] = np.asarray(handle[:], dtype=np.float32) / peaks[0]
        else:
            for j in range(0, dsize[2] ):
                x = io.imread(files[j], is_ome=False)
                vol4d[:, :, j+dsize[2], 0] = np.asarray(x, dtype=np.float32) / peaks[0]
        # Padding by 1st slice
        for j in range(0,dsize[2]):
            vol4d[:,:,j,0] = vol4d[:,:, dsize[2],0]

        print('Predicting...')
        for k in tqdm(range(0, origdim[2])):


            vol4dsplit = split_array(vol4d, psize, chunksize)  # vol4d is a 4D array HxWxDxC, dim2
            #print('Chunked image size = {}'.format(vol4dsplit.shape))  # vol4d becomes 5D array
            #sys.exit()
            adim = np.asarray(vol4dsplit.shape, dtype=int)
            #sys.exit()

            #adim2 = copy.deepcopy(adim[1:5])
            #adim2[0:3] = np.ceil(adim2[0:3] / 16) * 16  # Chunked images are already overlapped by psize, so no need to add padsize as psize again
            #print('Padded chunked image size = {}'.format(adim2))  # vol4d becomes 5D array
            #sys.exit()

            count = 0

            for p in range(0, chunksize[0]):
                for q in range(0, chunksize[1]):
                    # print('Chunk {} of {}:'.format(count+1,numsplit))
                    I1 = p * (origdim[0] // chunksize[0])
                    I2 = (p + 1) * (origdim[0] // chunksize[0])
                    J1 = q * (origdim[1] // chunksize[1])
                    J2 = (q + 1) * (origdim[1] // chunksize[1])

                    syn = ApplyModel3D(vol4dsplit[count, :, :, :, :], model, results.NETWORK)
                    #print(syn.shape)
                    #syn = syn[0:adim[1], 0:adim[2], psize[2]]
                    syn = syn[0:adim[1], 0:adim[2], dsize[2]]  # Since padding was done by dsize[2]
                    #print(syn.shape)
                    if output_is_file:
                        outvol[I1:I2, J1:J2, k] = syn[psize[0]:-psize[0] - 1, psize[1]:-psize[1] - 1]
                    else:
                        outvol[I1:I2, J1:J2] = syn[psize[0]:-psize[0] - 1, psize[1]:-psize[1] - 1]
                    syn = 0

                    if count == 0 and k == 0:  # Print only once
                        mem = nvmlDeviceGetMemoryInfo(gpuhandle)
                        print('Used = {} MB, Free = {} MB'.format(int(mem.used / (1024 ** 2)),
                                                                  int(mem.free / (1024 ** 2))))
                        print('Padded chunked image size = {}'.format(adim))  # vol4d becomes 5D array
                        #sys.exit()
                    count = count + 1

            if output_is_file == False:

                # outvol = peaks[0] * outvol/ (len(results.MODEL))
                outvol = peaks[0] * outvol
                if results.FLOAT == False:
                    outvol[outvol > 65535] = 65535
                    outvol = np.asarray(outvol, dtype=np.uint16)
                if 2 * np.prod(outvol.shape) >= 4 * (1024 ** 3):
                    io.imsave(outputfilenames[k], outvol, check_contrast=False, bigtiff=True, compression=results.COMPRESS)
                else:
                    io.imsave(outputfilenames[k], outvol, check_contrast=False, bigtiff=False, compression=results.COMPRESS)
            '''
            for j in range(0, 2 * psize[2]):
                vol4d[ :, :, j, :] = vol4d[ :, :, j+1, :]  # update current variable with previous variable, only read the last image, shift previous ones

            if os.path.isfile(results.IMAGES[0]):
                idx = k + psize[2] + 1
                # print(idx)
                if idx >= 0 and idx < origdim[2]:
                    handle.set_page(idx)
                    vol4d[ :, :, 2*psize[2],0] = np.asarray(handle[:], dtype=np.float32)
            else:
                idx = k + psize[2] + 1
                # print(idx)
                if idx >= 0 and idx < origdim[2]:
                    x = imread(files[idx], is_ome=False)
                    vol4d[:, :, 2*psize[2],0] = np.asarray(x, dtype=np.float32)
            '''
            for j in range(0, psize[2]-1):
                vol4d[:, :, j, :] = vol4d[:, :, j + 1, :]  # update current variable with previous variable, only read the last image, shift previous ones

            if os.path.isfile(results.IMAGES[0]):
                idx = k + dsize[2] + 1
                # print(idx)
                if idx >= 0 and idx < origdim[2]:
                    handle.set_page(idx)
                    vol4d[:, :, psize[2] - 1, 0] = np.asarray(handle[:], dtype=np.float32)

            else:
                idx = k + dsize[2] + 1
                # print(idx)
                if idx >= 0 and idx < origdim[2]:
                    x = imread(files[idx], is_ome=False)
                    vol4d[:, :, psize[2]-1, 0] = np.asarray(x, dtype=np.float32)

syn = 0
vol4d = 0
gc.collect()

if numsplit == 0:   # For small images, normalize it, for large stitched images, it is already normalized
    if modal[0].lower() == 'mic' or ext2 == '.tif' or ext2 == '.tiff':  # For large microscopy image, memory efficient scaling
                                                                        # If output is tif, we can safely assume, it is microscopy image
        print('Scaling by peak (= %.2f)' % (peaks[0]))
        for k in tqdm(range(0, origdim[2])):
            outvol[:, :, k] = peaks[0] * outvol[:, :, k]
            #outvol[:, :, k] = peaks[0]*outvol[:, :, k] / (len(results.MODEL))
            temp1 = outvol[:, :, k]
            if results.FLOAT == False:
                temp1[temp1 > 65535] = 65535
            outvol[:, :, k] = temp1
    else:   # If output is not TIF and the image is not microscopy, we can safely assume it is small, no need for memory efficient scaling
        #outvol = peaks[0]*outvol / (len(results.MODEL))  # Keep the original scaling, images will be float32
        if results.FLOAT == False:
            outvol = peaks[0] * outvol
            #outvol = peaks[0]*outvol / (len(results.MODEL))  # DImages are uint16, so likely images need to be rescaled


if modal[0].lower() == 'mic' and results.FLOAT == False: #save microscopy images as UINT16, unless --float overrides it
    outvol = np.asarray(outvol, dtype=np.uint16)



if numsplit == 0:  # outvol is saved in memory
    if results.FLOAT == False:
        outvol = np.asarray(outvol, dtype=np.uint16)
    if output_is_file==True:
        _, ext = os.path.splitext(os.path.basename(outname))
        print("Writing " + outname)
        if ext == '.nii' or ext == '.gz':
            obj = nifti.load(results.IMAGES[0])
            write_image(outvol, outname, niftiobj=obj)
        else:

            if 2*np.prod(outvol.shape)>= 4*(1024**3):
                write_image(outvol, outname, compression=results.COMPRESS, bigtifflag=True)
            else:
                write_image(outvol, outname, compression=results.COMPRESS, bigtifflag=False)

    else:
        outputfilenames = [None] * origdim[2]
        for i in range(0, origdim[2]):
            if input_is_file:
                s = os.path.basename(results.IMAGES[0])
                s, _ = os.path.splitext(s)
                s = s + '_Z' + str(i).zfill(4) + '.tif'
                s = os.path.join(outname, s)
                outputfilenames[i] = s
            else:
                s = os.path.basename(files[i])
                outputfilenames[i] = os.path.join(outname, s)
        print("Writing in " + outname)
        for k in tqdm(range(0, origdim[2])):
            io.imsave(outputfilenames[k], outvol[:, :, k], check_contrast=False, bigtiff=False, compression=results.COMPRESS)
else:
    if output_is_file == True:
        for k in range(0, origdim[2]):
            outvol[:, :, k] = peaks[0] * outvol[:, :, k]
            #outvol[:, :, k] = peaks[0] * outvol[:, :, k] / (len(results.MODEL))

        if results.FLOAT == False:
            for k in range(0, origdim[2]):
                temp1 = outvol[:, :, k]
                temp1[temp1 > 65535] = 65535
                outvol[:, :, k] = temp1

        if results.FLOAT == False:
            outvol = np.asarray(outvol, dtype=np.uint16)
        _, ext = os.path.splitext(os.path.basename(outname))
        print("Writing " + outname)
        if ext == '.nii' or ext == '.gz':  # Only if the input is nifti
            obj = nifti.load(results.IMAGES[0])
            write_image(outvol, outname, niftiobj=obj)
        else:

            if 2 * np.prod(outvol.shape) >= 4 * (1024 ** 3):
                write_image(outvol, outname, compression=results.COMPRESS, bigtifflag=True)
            else:
                write_image(outvol, outname, compression=results.COMPRESS, bigtifflag=False)
t2 = time.process_time()
print('Total time taken = {}'.format(datetime.timedelta(seconds=(t2 - t1))))
