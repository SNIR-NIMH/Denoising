import os, sys
import numpy as np
import argparse
from glob import  glob
from PIL import Image
import copy
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # This is critical, otherwise the nvidia-smi device order may not match with CUDA device order
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from skimage.io import imread,imsave
from tqdm import tqdm
import time
Image.MAX_IMAGE_PIXELS = 46340*46340
from CNNUtils.utils import dice_coeff, dice_coeff_loss, focal_loss, focal_loss_fixed, ssim_loss, ssim_metric
from keras.models import load_model
from argparse import RawDescriptionHelpFormatter
path = os.path.dirname(sys.argv[0])
path = os.path.abspath(path)
path = os.path.join(path, 'CNNUtils')
print('Appending {}'.format(path))
sys.path.append(path)




def split_array(input, psize, chunksize):  # Input is a 4D array now, HxWxDxC
    adim = np.asarray(input.shape, dtype=int)
    numsplit = np.prod(chunksize)
    sdim = (numsplit, adim[0] // chunksize[0] + 2 * psize[0] + 1, adim[1] // chunksize[1] + 2 * psize[1] + 1, adim[2])
    splitinput = np.zeros(sdim, dtype=np.float32)
    count = 0
    for i in range(0, chunksize[0]):
        for j in range(0, chunksize[1]):
            I1 = i * (adim[0] // chunksize[0]) - psize[0]
            I2 = (i + 1) * (adim[0] // chunksize[0]) + psize[0] + 1
            J1 = j * (adim[1] // chunksize[1]) - psize[1]
            J2 = (j + 1) * (adim[1] // chunksize[1]) + psize[1] + 1
            delta = [0, 0]
            if I1 < 0:
                delta[0] = -I1
                I1 = 0
            if I2 > adim[0]:
                I2 = adim[0]
            if J1 < 0:
                delta[1] = -J1
                J1 = 0
            if J2 > adim[1]:
                J2 = adim[1]

            x = input[I1:I2, J1:J2, :]

            splitinput[count, delta[0]:x.shape[0] + delta[0], delta[1]:x.shape[1] + delta[1], :] = x
            count = count + 1

    return splitinput  # It is numsplit x dH x dW x D


def ApplyModel3D(in_vol, model, network):  # in_vol is HxWxD matrix, this function will NOT pad what it gets and runs the model

    dim = in_vol.shape  # 3D array

    if str(network).lower() == 'unet' or str(network).lower() == 'attentionunet' \
        or str(network).lower() == 'unet++' or str(network).lower() == 'inception':  # The inception architecture is unet-like
        adim = np.ceil(np.asarray(dim[0:3], dtype=float) / 16) * 16
        # All Unets need dimensions multiple of 16
    else:
        adim = copy.deepcopy(dim)

    batch_dim = np.asarray((1, adim[0], adim[1], adim[2], 1), dtype=int)  # Make is 5D, training was on 5D array
    batch = np.zeros(batch_dim, dtype=np.float32)

    batch[0, 0:dim[0], 0:dim[1], 0:dim[2], 0] = in_vol
    #print(batch.shape)
    pred = model.predict_on_batch(batch)
    out_vol = pred[0, 0:dim[0], 0:dim[1], 0:dim[2], 0]

    return out_vol

usage = '''Example:

Step1: Create appropriate symlinks:
python  Denoise_Test3D_multigpu.py --func prepare -i /home/user/input_folder/  -o /home/user/output_folder/ 
                            --psize 64 64 64  --model /home/user/my3Dmodel.h5 --n 52 --network unet --gpu 0 --chunks 9 12 

It will create N folders (N=number of slices) with appropriate symlinks and also create a swarm file. 

Step2: The swarm file contains the same command except --func run and inputs as those N folders. For a cluster, swarm the 
file. For single-node multi-GPU system, simply change the GPU ids from default 0 to the number of GPUs and parallelize 
the script via ppss or GNU-parallel.
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D model prediction  with multiple GPUs', formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('--func', type=str, dest='FUNCTION', default=None, required=True,
                        help='Two options, prepare or run. (1) First step is to prepare appropriate data by creating '
                             'multiple symlinks for each slice in a separate folder. '
                             '(2) After appropriate symlinks are created, run each slice separately on a GPU. ')
    parser.add_argument('--im','-i', required=True, dest='INPUT', type=str,
                        help='Input directory containing multiple 2D tif image slices. 3D tifs are not accepted. ')

    parser.add_argument('--o', '-o', required=True, action='store', dest='OUTPUT',
                        help='Output folder, e.g. /home/user/output_folder/ where 2D slices will be written.')
    parser.add_argument('--model', required=True, dest='MODEL', type=str,
                        help='Trained model (.h5) file.')
    parser.add_argument('--psize', required=True, type=int, nargs='+', dest='PATCHSIZE',
                        help='Same 3D patch size used for training. If network is UNET or variants of UNET, patch sizes '
                             'must be multiple of 16.')


    # Optional inputs
    parser.add_argument('--n', required=False, type=int, dest='NUMGPU', default=10,
                        help='Number of parallel GPUs to use, default 10. **** Note: This only works for --func=prepare.')
    parser.add_argument('--network', required=False, dest='NETWORK', type=str, default='UNET',
                        help='Type of network used for training. Options are Unet, DenseNet, Inception, RCAN, UNET++, EDSR, AttentionUnet')
    parser.add_argument('--gpu', required=False, action='store', dest='GPU', type=int, default=0,
                        help='GPU id to use on each node. Default is 0.')
    parser.add_argument('--chunks', required=False, dest='CHUNKS', type=int, nargs='+', default=[1,1],
                        help='If the input image size is too large (such as stitched images) to fit into GPU memory, '
                             'it can be chunked using "--chunks nh nw" argument. E.g. --chunks 3 2 will split a '
                             'HxWxD image into overlapping (H/3)x(W/2)xD chunks, apply the trained models on '
                             'each chunk serially, then join the chunks. ')

    parser.add_argument('--float', required=False, dest='FLOAT', action='store_true',
                        help='Use this option to save output images as FLOAT32. Default is UINT16. Note, saving as FLOAT32 images will '
                             'double the size of the output image.')
    parser.epilog = usage

    results = parser.parse_args()
    code = sys.argv[0]
    if len(results.PATCHSIZE) != 3:
        sys.exit('ERROR: Patch must be 3D. For 2D patches, use Denoise_Test2D_multigpu.py')
    if str(results.FUNCTION).lower() not in ['prepare', 'run']:
        sys.exit('ERROR: Function must be prepare or run.')

    results.OUTPUT = os.path.realpath(os.path.expanduser(results.OUTPUT))  # This is absolutely important because of symlinks and tilde
    results.MODEL = os.path.realpath(os.path.expanduser(results.MODEL))

    if os.path.isdir(results.INPUT) == False:
        sys.exit('ERROR: Input must be a folder containing multiple 2D slices.')

    if os.path.isfile(results.OUTPUT) == True:
        print('ERROR: The output must be a folder. {}'.format(results.OUTPUT))
        sys.exit()

    if results.NUMGPU <= 1:
        print('ERROR: Enter more than 1 GPUs via --n argument. You have entered {}'.format(results.NUMGPU))
        sys.exit()

    os.makedirs(results.OUTPUT, exist_ok=True)
    inputfilelist = sorted(glob(os.path.join(results.INPUT,'*.tif')))

    psize = results.PATCHSIZE
    uid = time.strftime('%d-%m-%Y_%H-%M-%S')
    uid = 'tempdir_' + uid

    try:
        x = Image.open(inputfilelist[-1])
    except:
        try:
            x = imread(inputfilelist[-1], is_ome=False)
        except:
            sys.exit('ERROR: Input files can not be read by Pillow or Scikit-image.')

    x = np.asarray(x, dtype=np.float32)
    dim = (x.shape[0], x.shape[1], len(inputfilelist))




    if str(results.FUNCTION).lower() == 'prepare':
        #
        # for 3D patches, create N folders for N images. Each folder contains symlinks to each slice along with its
        # psize[3]/2 neighbors. Then run a swarm on each folder.
        N = len(inputfilelist)
        print('Creating symlinks for {} slices.'.format(N))
        os.makedirs(os.path.join(results.OUTPUT, uid), exist_ok=True)
        outputdir = os.path.join(results.OUTPUT, uid)

        for i in tqdm(range(N)):
            s1 = os.path.join(outputdir, str(i).zfill(4))
            os.makedirs(s1, exist_ok=True)
            count =0
            for j in range(-psize[2]//2, psize[2]//2):  # This confirms that the number of slices will always be multiple of 16
                if i+j<0:
                    fname = inputfilelist[0]
                elif i+j>=N:
                    fname = inputfilelist[N-1]
                else:
                    fname = inputfilelist[i+j]
                s2 = os.path.realpath(os.path.abspath(os.path.expanduser(fname)))
                s3 = 'Z' + str(count).zfill(4) + '.tif'
                s3 = os.path.join(s1, s3)
                os.symlink(s2, s3)
                count=count+1

        s = 'swarm_' + uid + '.swarm'
        s = os.path.join(outputdir, s)
        s = os.path.realpath(os.path.expanduser(s))
        f1 = open(s, 'w+')
        for i in range(N):
            outputdir2 = os.path.join(outputdir, str(i).zfill(4))

            if results.FLOAT == True:
                floatadd = ' --float'
            else:
                floatadd = ''
            print('python %s --func run --im %s --model %s --o %s --gpu %d --chunks %d %d --psize %d %d %d --network %s %s'
                    % (code, outputdir2, results.MODEL, results.OUTPUT, results.GPU, results.CHUNKS[0],
                       results.CHUNKS[1], results.PATCHSIZE[0], results.PATCHSIZE[1], results.PATCHSIZE[2], results.NETWORK, floatadd), file=f1)

        f1.close()
        mem = (4 * 4 * dim[0]*dim[1]*psize[2] ) / (1024.0 ** 3)
        # max memory is 4 float32 variables of the 3D image size (??? could be 2)
        mem = np.ceil(1.1*mem / 10) * 10  # Assuming 10% overhead
        b = np.ceil(N/results.NUMGPU)
        print('Maximum memory required = %d GB' % (mem))
        print(':========================================================')
        print('Now run this on a Biowulf login node shell:')
        print('swarm -f %s --partition=gpu --merge-output --gres=gpu:k80:1  -t 4 -g %d  --time 1:00:00 -b %d' % (s, mem,b))
        print(':========================================================')
        print('*** NOTE: The estimated memory (%d GB) is really an estimate. It is strongly advised to run one command '
              'of the swarm first to check the appropriate memory requirement, chunksize, and runtime' %(mem))


    else:

        if os.getenv('CUDA_VISIBLE_DEVICES') is None or len(os.getenv('CUDA_VISIBLE_DEVICES')) == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(results.GPU)
            print('Setting CUDA_VISIBLE_DEVICES to {}'.format(os.getenv('CUDA_VISIBLE_DEVICES')))
        else:
            print('SLURM already sets GPU id to {}, I will not change it.'.format(os.getenv('CUDA_VISIBLE_DEVICES')))
            # results.GPU = int(os.getenv('CUDA_VISIBLE_DEVICES').split(',')[0])

        import tensorflow as tf

        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        psize = np.asarray(results.PATCHSIZE, dtype=int)
        chunksize = np.asarray(results.CHUNKS, dtype=int)

        print('Reading the whole image into memory: {}'.format(results.INPUT))
        if os.path.isdir(results.INPUT):

            s = os.path.join(results.INPUT, '*.tif')
            files = sorted(glob(s))
            if len(files) == 0:  # .tif files not found
                s = os.path.join(results.INPUTDIR, '*.tiff')
                files = sorted(glob(s))

            if len(files) == 0:
                print('ERROR: .tif or .tiff files not found in {}'.format(results.IMAGES[0]))
                sys.exit()
            try:
                # print('Reading image with PIL.Image')
                x = Image.open(files[-1])
            except:
                try:
                    # print('Reading image with skimage.io.imread')
                    # print(files[-1])
                    x = imread(files[-1])
                except:
                    sys.exit('ERROR: Image can not be read with PIL.Image or skimage.io.imread')

            x = np.asarray(x, dtype=np.uint16)
            origdim = x.shape
            origdim = (origdim[0], origdim[1], len(files))
            vol4d = np.empty(origdim, dtype=np.float32)

            for k in range(origdim[2]):
                x = imread(files[k], is_ome=False)
                x = np.asarray(x, dtype=np.float32)
                vol4d[:, :, k] = x

        elif os.path.isfile(results.INPUT):
            vol4d = imread(results.INPUT, is_ome=False)
            origdim = vol4d.shape

        print('Image dimension = {}'.format(origdim))
        outvol = np.zeros((origdim[0], origdim[1]), dtype=np.float32)

        dict = {"tf": tf,
                "dice_coeff": dice_coeff,
                "dice_coeff_loss": dice_coeff_loss,
                "focal_loss": focal_loss,
                "focal_loss_fixed": focal_loss_fixed,
                "ssim_loss": ssim_loss,
                "ssim_metric": ssim_metric,

                }

        model = load_model(results.MODEL, custom_objects=dict)

        print('Predicting...')

        vol4d = split_array(vol4d, psize, chunksize)  # vol4d is a 4D array HxWxDxC, dim2
        # vol4dsplit = vol4dsplit[:,:,:, :, np.newaxis] # It is now 5D, batchsizexHxWxDxC, C=1
        print('Chunked image size = {}'.format(vol4d.shape))  # vol4d becomes 5D array

        adim = np.asarray(vol4d.shape, dtype=int)

        count = 0

        for p in tqdm(range(0, chunksize[0]),desc='H chunks'):
            for q in tqdm(range(0, chunksize[1]),desc='W chunks', leave=False):
                # print('Chunk {} of {}:'.format(count+1,numsplit))
                I1 = p * (origdim[0] // chunksize[0])
                I2 = (p + 1) * (origdim[0] // chunksize[0])
                J1 = q * (origdim[1] // chunksize[1])
                J2 = (q + 1) * (origdim[1] // chunksize[1])

                syn = ApplyModel3D(vol4d[count, :, :, :], model, results.NETWORK)
                syn = syn[:,:, psize[2]//2]

                outvol[I1:I2, J1:J2] = syn[psize[0]:-psize[0] - 1, psize[1]:-psize[1] - 1]
                count = count + 1

        #outvol = outvol[:,:,psize[2]//2]
        outname = inputfilelist[psize[2]//2]
        outname = os.path.basename(os.path.realpath(outname))
        outname, _ = os.path.splitext(outname)
        outname = outname + '_denoised.tif'
        outname = os.path.join(results.OUTPUT, outname)
        print('Writing to {}'.format(outname))
        if results.FLOAT == False:
            outvol[outvol > 65535] = 65535
            outvol = np.asarray(outvol, dtype=np.uint16)
        if outvol.nbytes >= 4*(1024**3):
            btif = True
        else:
            btif = False
        a = outvol.nbytes / (1024 ** 3)
        print('Writing {}'.format(outname))
        imsave(outname, outvol, check_contrast=False, bigtiff=btif, compression='zlib')
