import os, sys
import numpy as np
import argparse
from glob import  glob
from PIL import Image
import tempfile
from tqdm import tqdm
import time
from skimage.io import  imread
Image.MAX_IMAGE_PIXELS = 46340*46340



path = os.path.dirname(sys.argv[0])
path = os.path.abspath(path)
code = os.path.join(path, 'Denoise_Test.py')

if os.path.isfile(code) ==False:
    print('ERROR: Denoise_Test.py code not found in {}'.format(code))
    print('ERROR: Please keep this code in the same path as Denoise_Test.py')
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2D model Prediction  with multiple GPUs')


    parser.add_argument('--im', required=True, dest='IMAGES', type=str,
                        help='Input image, saved as a single directory containing multiple 2D tif image slices. ')

    parser.add_argument('--o', required=True, action='store', dest='OUTPUT',
                        help='Output folder, e.g. /home/user/output_folder/ '
                             'where 2D slices will be written.')
    parser.add_argument('--model', required=True, dest='MODEL', type=str,
                        help='Trained models (.h5) files. ')
    parser.add_argument('--psize', required=True, type=int, nargs='+', dest='PATCHSIZE',
                        help='Same 2D patch size used for training.')
    parser.add_argument('--n', required=True, type=int, dest='NUMGPU', default=0,
                        help='Number of parallel GPUs to use. **** Note: This only works for 2D models.')
    parser.add_argument('--network', required=True, dest='NETWORK', type=str,
                        help='Type of network used for training. Options are Unet, DenseNet, Inception, RCAN, UNET++, EDSR, AttentionUnet')
    # Optional inputs

    parser.add_argument('--gpu', required=False, action='store', dest='GPU', type=int, default=0,
                        help='GPU id to use on each node. Default is 0.')
    parser.add_argument('--chunks', required=False, dest='CHUNKS', type=int, nargs='+', default=[1,1],
                        help='If the input image size is too large (such as stitched images) to fit into GPU memory, '
                             'it can be chunked using "--chunks nh nw" argument. E.g. --chunks 3 2 will split a '
                             'HxWxD image into overlapping (H/3)x(W/2)xD chunks, apply the trained models on '
                             'each chunk serially, then join the chunks. This option works only if (1) the input and '
                             'outout images are both TIF (either 3D or a folder), (2) only one channel is available. ')

    parser.add_argument('--float', required=False, dest='FLOAT', action='store_true',
                        help='Use this option to save output images as FLOAT32. Default is UINT16. This is useful '
                             'if the dynamic range of the training data is small. Note, saving as FLOAT32 images will '
                             'double the size of the output image.')

    results = parser.parse_args()

    results.OUTPUT = os.path.realpath(os.path.expanduser(results.OUTPUT))  # This is absolutely important because of symlinks and tilde
    results.MODEL = os.path.realpath(os.path.expanduser(results.MODEL))

    if os.path.isdir(results.IMAGES) == False:
        sys.exit('ERROR: Input must be a folder containing multiple 2D slices.')

    if os.path.isfile(results.OUTPUT) == True:
        print('ERROR: The output must be a folder. {}'.format(results.OUTPUT))
        sys.exit()

    if results.NUMGPU <= 1:
        print('ERROR: Enter more than 1 GPUs via --n argument. You have entered {}'.format(results.NUMGPU))
        sys.exit()

    os.makedirs(results.OUTPUT, exist_ok=True)
    inputfilelist = sorted(glob(os.path.join(results.IMAGES,'*.tif')))
    N = len(inputfilelist)
    if N < results.NUMGPU:
        results.NUMGPU = N
    N = N//results.NUMGPU


    print('%d images found. They will be split into %d GPUs in %d counts.' %(len(inputfilelist), results.NUMGPU, N))
    psize = results.PATCHSIZE
    if len(psize) != 2:
        sys.exit('ERROR: Patch must be 2D. For 3D patches, use Synthesis_3DTest_multigpu.py')
    uid = time.strftime('%d-%m-%Y_%H-%M-%S')

    try:
        x = Image.open(inputfilelist[-1])
    except:
        try:
            x = imread(inputfilelist[-1], is_ome=False)
        except:
            sys.exit('ERRO: Input files can not be read by Pillow or Scikit-image.')

    x = np.asarray(x, dtype=np.float32)
    dim = x.shape

    if len(psize) == 2:  # For 2D patches, create M = num_images/num_gpu folders and run Synthesis_Test.py
        # on each of the folder. Synthesis_Test.py can take a folder. It is possible to run it for each image, but running
        # on a folder is much more efficient on a cluster because once GPUs are allocated, they will be used always, rather
        # than waiting for a new GPU once on slice is finished
        N = len(inputfilelist)
        if N < results.NUMGPU:
            results.NUMGPU = N
        N = N // results.NUMGPU
        print('%d images found. They will be split into %d GPUs in %d counts.' % (len(inputfilelist), results.NUMGPU, N))

        mem = 4*4*np.prod(dim)/(1024.0**3)  # max memory is 4 float32 variables of the 2D image size
        mem = np.ceil(1.1*mem/10)*10        # Assuming 10% overhead
        print('Maximum memory required = %d GB' %(mem))

        os.makedirs(os.path.join(results.OUTPUT,uid), exist_ok=True)
        outputdir = os.path.join(results.OUTPUT,uid)
        for i in range(0,results.NUMGPU):
            os.makedirs(os.path.join(outputdir, str(i).zfill(3)), exist_ok=True)

        count=0
        for i in range(0, results.NUMGPU):
            outputdir2 = os.path.join(outputdir, str(i).zfill(3))
            for j in range(0,N):
                s1 = os.path.realpath(os.path.abspath(os.path.expanduser(inputfilelist[count+j])))
                s2 = os.path.basename(s1)
                s2 = os.path.join(outputdir2, s2)
                os.symlink(s1,s2)

            count = count+N

        if count<len(inputfilelist):
            i=0
            for j in range(count,len(inputfilelist)):
                outputdir2 = os.path.join(outputdir, str(i).zfill(3))
                s1 = os.path.realpath(os.path.expanduser(inputfilelist[j]))
                s2 = os.path.basename(s1)
                s2 = os.path.join(outputdir2, s2)
                os.symlink(s1, s2)
                i=i+1

        s = 'swarm_' + uid + '.swarm'
        s = os.path.join(outputdir, s)
        s = os.path.realpath(os.path.expanduser(s))
        f1 = open(s, 'w+')
        for i in range(0, results.NUMGPU):
            outputdir2 = os.path.join(outputdir, str(i).zfill(3))
            if results.FLOAT == True:
                print(
                    'python %s --im %s --model %s --modalities unk --o %s --gpu %d --chunks %d %d --psize %d %d --float --network %s'
                    % (code, outputdir2, results.MODEL, results.OUTPUT, results.GPU, results.CHUNKS[0],
                       results.CHUNKS[1], results.PATCHSIZE[0], results.PATCHSIZE[1], results.NETWORK), file=f1)
            else:
                print(
                    'python %s --im %s --model %s --modalities unk --o %s --gpu %d --chunks %d %d --psize %d %d  --network %s'
                    % (code, outputdir2, results.MODEL, results.OUTPUT, results.GPU, results.CHUNKS[0],
                       results.CHUNKS[1], results.PATCHSIZE[0], results.PATCHSIZE[1], results.NETWORK), file=f1)

        f1.close()
        print(':============================')
        print('Now run this on a Biowulf login node shell:')
        print('swarm -f %s --partition=gpu --merge-output --gres=gpu:k80:1  -t 4 -g %d  --time 8:00:00' % (s, mem))
        print(':============================')
