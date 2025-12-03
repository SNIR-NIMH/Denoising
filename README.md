<h1 align="center">Terabyte scale image denoising using multiple nodes & multiple GPUs</h1>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>      
    </li>
   <li>
      <a href="#overview">Overview</a>      
    </li>
   <li>
      <a href="#installation">Installation</a>      
    </li>
    <li>
      <a href="#training">Training</a>      
    </li>
    <li>
      <a href="#prediction">Prediction</a>  
      <ul>
        <li><a href="#multi-gpu-multi-node-prediction">Multi-GPU multi-node prediction</a></li>
        <li><a href="#prediction-with-2d-patches">Prediction with 2D patches</a></li>
         <li><a href="#prediction-with-3d-patches">Prediction with 3D patches</a></li>
      </ul>
    </li>
    
    
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains Python based image denoising scripts using common CNN
architectures like Unet and its variations, RCAN, Inception etc. The primary focus
is to train a model from a limited training dataset (e.g. 3-5) and apply it to
very large multi-Terabyte 3D images (e.g. 100k x 100k pixels with thousands of slices)
using multiple GPUs on multiple nodes. A simple GUI is also provided to train/predict
small images.



## Overview
The following figure shows an example of a 3D noisy and denoised image. It has
been denoised using a 3D UNET using 64x64x16 patch size.

Noisy Image             |  Denoised Image
:-------------------------:|:-------------------------:
![](figs/ch00-1.gif)  |  ![](figs/ch00_denoised3D-1.gif)

## Installation 

1. Libtiff is required. If root access is available, install
   libtiff-devel (```sudo yum install libtiff-devel```). If root access is
   not available, download the source from ```https://download.osgeo.org/libtiff/```. The
scripts are tested with libtiff 4.6.0 with GCC-11.3.0. Download tiff-4.6.0.zip, unzip, compile, and
After installing to a suitable location, add the lib folder to LD_LIBRARY_PATH and
the include folder to CPATH, e.g.,
```
export LD_LIBRARY_PATH=/home/user/libtiff/4.6.0/install/lib:${LD_LIBRARY_PATH}
export CPATH=/home/user/libtiff/4.6.0/install/include:${CPATH}
```
It is strongly recommended to compile with DEFLATE support (libdeflate) to ensure reading
compressed 3D tifs.

2. Create new python environment with tensorflow-gpu,
```
conda create -n myowncondaenv python==3.10
conda activate myowncondaenv
pip install tensorflow-gpu==2.11.0
pip install cython==0.29.36
pip install pytiff
pip install scikit-image==0.19.3
```
Then ```python Denoise_Train.py -h``` should show the help.


## Training
Multi-GPU single node training can be done with 2D or 3D patches depending on the application,
```
python Denoise_Train.py -h
Appending /home/user/SNIR-NIMH-Denoising/CNNUtils
usage: Denoise_Train.py [-h] --atlasdir ATLASDIR --natlas NUMATLAS --psize PATCHSIZE [PATCHSIZE ...] --model MODEL --o OUTDIR [--modalities MODAL] [--gpu GPU]
                        [--maxpatch MAXPATCH] [--basefilters BASEFILTER] [--batchsize BATCHSIZE] [--epoch EPOCH] [--loss LOSS] [--initmodel INITMODEL] [--lr LR]

Image Denoising Training.

options:
  -h, --help            show this help message and exit

Required arguments:
  --atlasdir ATLASDIR   Atlas directory should contain atlas{X}_M1.tif, atlas{X}_M2.tif, atlas{X}_GT.tif, X=1,2,3.. etc. The M1, M2 etc denote 1st, 2nd
                        modalities and GT denotes ground truth. See --modalities. All patches with non-zero center voxels will be considered for training.
                        Optionally, if atlas{X}_mask.tif binary images are present, then patches will be collected from the non-zero indices of the
                        atlas{X}_mask.tif images.
  --natlas NUMATLAS     Number of atlases to be used. Atlas directory must contain at least these many atlases.
  --psize PATCHSIZE [PATCHSIZE ...]
                        2D or 3D patch size, e.g. --psize 256 256 or --psize 32 32 32. **** Patch sizes must be multiple of 16.****
  --model MODEL         Training model, options are Unet, DenseNet, Inception, RCAN, UNET++, EDSR, AttentionUnet
  --o OUTDIR            Output directory where the trained models are written.

Optional arguments:
  --modalities MODAL    A string of input and target image modalities. Accepted modalities are T1/T2/PD/FL/CT/UNK. Default unk,unk,..., i.e. no normalization.
                        Normally microscopy images dont need normalization.
  --gpu GPU             GPU id or ids to use for training. Example --gpu 1 indicates gpu with id 1 will be used for training. For multi-gpu training, use comma
                        separated list, such as --gpu 2,3,4.
  --maxpatch MAXPATCH   Maximum number of patches to be collected from each atlas. Default is 50,000. Generally 100,000 patches are good enough. As a thumb
                        rule, use 100000/natlas.
  --basefilters BASEFILTER
                        Base number of convolution filters to be used. Usually 8-32 works well. For UNET, maximum number of filters in last conv block in Unet
                        is 16 x BASEFILTERS.
  --batchsize BATCHSIZE
                        Batch size. Default 64. Usually 32-96 works well. Decrease for 3D patches or large 2D patches.
  --epoch EPOCH         Maximum number of epochs to run. Default is 20. Usually 50-100 works well.
  --loss LOSS           Loss type. either MSE or MAE. Default MAE.
  --initmodel INITMODEL
                        Pre-trained model to initialize, if available.
  --lr LR               Learning rate for Adam optimizer. Default is 0.0001.
```


Example 2D or 3D training usage,
```
python Denoise_Train.py --atlasdir /home/user/denoise_atlas/  --natlas 6 --psize 256 256 \
  --model unet --o /home/user/denoise_atlas/ --basefilter 32 --loss mae --batchsize 64 \
  --epoch 50 --gpu 0,1,2,3 --maxpatch 15000
or
python Denoise_Train.py --atlasdir /home/user/denoise_atlas/  --natlas 6 --psize 64 64 16 \
 --model unet --o /home/user/denoise_atlas/ --basefilter 32 --loss mae --batchsize 16 \
 --epoch 50 --gpu 0,1,2,3 --maxpatch 5000
```

Training GT, i.e. ground truth noise-free images are usually obtained by acquiring same
ROI 200 or more times and averaging them as 32-bit images. Then atlas{X}_M1 can be any one
of the noisy image.

## Prediction
The trained model can be applied to any size image, either a 3D tif or a folder containing 
multiple 2D tifs. 

```
python Denoise_Test.py -h
Appending /home/user/SNIR-NIMH-Denoising/CNNUtils
usage: Denoise_Test.py [-h] --im IMAGES [IMAGES ...] --o OUTPUT --model MODEL --network NETWORK --psize PATCHSIZE [PATCHSIZE ...] [--modalities MODAL]
                       [--gpu GPU] [--chunks CHUNKS [CHUNKS ...]] [--float] [--compress]

Model prediction on single GPU

options:
  -h, --help            show this help message and exit
  --im IMAGES [IMAGES ...], -i IMAGES [IMAGES ...]
                        Input image(s), nifti (.nii or .nii.gz) or TIF (.tif or .tiff). The order must be same as the order of atlas{X}_M1.nii.gz,
                        atlas{X}_M2.nii.gz images, i.e. 1stinput must be of channel M1, second M2, etc. For microscopy images,a single directory containing
                        multiple 2D tif image slices is acceptable.
  --o OUTPUT, -o OUTPUT
                        Output filename, e.g. somefile.nii.gz or somefile.tif where the result will be written. If the image is large, use a folder as output,
                        e.g. /home/user/output_folder/ where 2D slices will be written. Output can be NIFTI only if the input is also NIFTI.
  --model MODEL         Trained model (.h5) files.
  --network NETWORK     Type of the network used for training. Options are Unet, DenseNet, Inception, RCAN, UNET++, EDSR, AttentionUnet
  --psize PATCHSIZE [PATCHSIZE ...]
                        Same 2D or 3D patch size used for training.
  --modalities MODAL    (Optional) A comma separated string of input image modalities. Accepted modalities are T1/T2/PD/FL/CT/MIC/UNK. Default is unk,unk. This
                        is the same as entered during training. Default: If images are not needed to be normalized (same for training), then use UNK (unknown)
                        as modality.
  --gpu GPU             (Optional) GPU id to use. Default is 0.
  --chunks CHUNKS [CHUNKS ...]
                        (Optional) If the input image size is too large to fit into GPU memory, it can be chunked using "--chunks nh nw" argument. E.g. --chunks
                        3 2 will split a HxWxD image into overlapping (H/3)x(W/2)xD chunks, apply the trained models on each chunk serially, then join the
                        chunks. This option works only if (1) the input and outout images are both TIF (either 3D or a folder), (2) only one channel is
                        available. **Normally, if image is not chunked, total memory required is 6 times the size of the image.** Default: no chunking
  --float               (Optional) Use --float to save output images as FLOAT32. Default is UINT16. This is useful if the dynamic range of the training data is
                        small. Note, saving as FLOAT32 images will double the size of the output image.
  --compress            (Optional) If --compress is used, the output Tif images will be compressed.
```



Example usage:
```
python Denoise_Test.py --im /home/user/small_tif_image.tif --model mymodel_128x128_UNET_UNK+UNK.h5 \
         --o /home/user/denoised_image.tif --gpu 0 --psize 128 128 --network UNET
or
python Denoise_Test.py --im /home/user/large_3D_image.tif --model mymodel_128x128_UNET_UNK+UNK.h5  \
         --o /home/user/denoised_tifs/ --gpu 0 --chunks 3 2 --psize 128 128 --network UNET
or
python Denoise_Test.py --im /home/user/folder_with_large_2Dtifs/  --model mymodel_64x64x16_UNET_UNK+UNK.h5 \
         --o /home/user/denoised_tifs/ --gpu 0 --chunks 20 40 --psize 64 64 16 --network UNET
```

### Multi-GPU multi-node prediction

Very large multi-Terabyte images can be denoised using multiple GPUs and multiple nodes using Denoise_Test2D_multigpu.py
or Denoise_Test3D_multigpu.py, for 2D or 3D patches, respectively.
For this script, 
* the input image must be saved as 2D tifs in a folder (3D tifs are not acceptable)
* output must be a folder,
* only single channel images are accepted,
  
### Prediction with 2D patches

Usage:
```
python Denoise_Test2D_multigpu.py -h
usage: Denoise_Test2D_multigpu.py [-h] --im IMAGES --o OUTPUT --model MODEL --psize PATCHSIZE [PATCHSIZE ...] --n NUMGPU --network NETWORK [--gpu GPU]
                                [--chunks CHUNKS [CHUNKS ...]] [--float]

Model Prediction with multiple GPUs

options:
  -h, --help            show this help message and exit
  --im IMAGES           Input image, saved as a single directory containing multiple 2D tif image slices.
  --o OUTPUT            Output folder, e.g. /home/user/output_folder/ where 2D slices will be written.
  --model MODEL         Trained models (.h5) files.
  --psize PATCHSIZE [PATCHSIZE ...]
                        Same 2D or 3D patch size used for training.
  --n NUMGPU            Number of parallel GPUs to use. **** Note: This only works for 2D models.
  --network NETWORK     Type of network used for training. Options are Unet, DenseNet, Inception, RCAN, UNET++, EDSR, AttentionUnet
  --gpu GPU             GPU id to use on each node. Default is 0.
  --chunks CHUNKS [CHUNKS ...]
                        If the input image size is too large (such as stitched images) to fit into GPU memory, it can be chunked using "--chunks nh nw"
                        argument. E.g. --chunks 3 2 will split a HxWxD image into overlapping (H/3)x(W/2)xD chunks, apply the trained models on each chunk
                        serially, then join the chunks. This option works only if (1) the input and outout images are both TIF (either 3D or a folder), (2) only
                        one channel is available.
  --float               Use this option to save output images as FLOAT32. Default is UINT16. This is useful if the dynamic range of the training data is small.
                        Note, saving as FLOAT32 images will double the size of the output image.
```

Example usage:
```
python Denoise_Test_multigpu.py --im /home/user/folder_with_large_2Dtifs/  --model mymodel_256x256_UNET_UNK+UNK.h5 \
         --o /home/user/denoised_tifs/ --gpu 0 --chunks 20 40 --psize 256 256 --network UNET --n 46 --float
```
This script will split the input image into N (#GPUs) folders and create a swarm file (a text file)
calling Denoise_Test.py to process each folder independently. The swarm file can be swarmed in a cluster.

For a multi-GPU single node system, simply run the script with ```--n N, where N=number of available GPUs```, 
and then edit the output swarm file from ```--gpu 0``` to ```--gpu 1```, ```--gpu 2``` etc. Then use GNU Parallel
or PPSS to run all of the N lines together.

### Prediction with 3D patches

Prediction with 3D patches is a 2-step processes. First, for each of the N slices in a folder, N subfolders are
created, where i-th subfolder contains symbolic links of P slices around the i-th slice, P being the patch size in Z.
Then a 3D prediction on each of those N subfolders are obtained in parallel or in cluster.

Step1: Create appropriate symlinks:
```
python  Denoise_Test3D_multigpu.py --func prepare -i /home/user/input_folder/  -o /home/user/output_folder/ 
                            --psize 64 64 64  --model /home/user/my3Dmodel.h5 --n 52 --network unet --gpu 0 --chunks 9 12 
```
It will create N folders (N=number of slices) with appropriate symlinks and also create a swarm file. 

Step2: The swarm file contains the same command except --func run and inputs as those N folders. For a cluster, swarm the 
file. For single-node multi-GPU system, simply change the GPU ids from default 0 to the number of GPUs and parallelize
