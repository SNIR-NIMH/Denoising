import numpy as np
from scipy.ndimage import  rotate
import random
from scipy.ndimage import zoom
import cv2
import os, sys
import copy
import time
from datetime import datetime

def adjust_gamma(image, gamma=1.0):

    image2 = image/65535 # 16bit image by default, could be changed later
    image2 = image2 ** gamma
    image2 = image2*65535
    return image2


def cv2_clipped_zoom(img, zoom_factor=0):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    ------
    Args:
        img : ndarray
            Image array
        zoom_factor : float
            amount of zoom as a ratio [0 to Inf). Default 0.
    ------
    Returns:
        result: ndarray
           numpy ndarray of the same shape of the input img zoomed by the specified factor.
    """
    if zoom_factor == 0:
        return img

    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height),0,0,interpolation=cv2.INTER_NEAREST)
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result



def augmentor2d(inputimg, inputseg, prob=0.5):  # Apply augmentation to randomly chosen 50% of the sample

    #rng = random.SystemRandom()

    N = inputimg.shape[0]  # for 2D patches, input is NxHxWxC, N=number of patches
    N1 = int(np.round(N * prob))
    dim = np.asarray(inputimg.shape, dtype=int)
    dim2 = copy.deepcopy(dim)
    dim2[0] = dim2[0] + N1*5 # 5 augmentations
    outputimg = np.zeros(dim2, dtype=np.float32)
    outputseg = np.zeros(dim2, dtype=np.float32)

    outputimg[0:dim[0], :, :, :] = inputimg
    outputseg[0:dim[0], :, :, :] = inputseg

    # Up down flipping
    print('Up-down flipping')
    list = np.asarray(range(0, N), dtype=int)
    np.random.seed(int(datetime.now().timestamp()))
    np.random.shuffle(list)
    list = list[0: N1]
    input2 = inputimg[list, :, :, :]
    input2s = inputseg[list, :, :, :]
    #dim = np.asarray(input2.shape, dtype=int)# dim is 4x1, NxHxWxC, N= number of patches
    #input3 = np.zeros(dim, dtype=np.float32)
    #input3s = np.zeros(dim, dtype=np.float32)
    count = N
    for i in range(0,N1):
        for c in range(0,dim[3]):
            outputimg[count+i, :, :, c] = np.flipud(input2[i, :, :, c])
            outputseg[count+i, :, :, c] = np.flipud(input2s[i, :, :, c])
    #outputimg = np.concatenate((inputimg, input3), axis=0)
    #outputseg = np.concatenate((inputseg, input3s), axis=0)

    # Left right flipping
    print('Left-right flipping')
    list = np.asarray(range(0, N), dtype=int)
    np.random.seed(int(datetime.now().timestamp()))
    np.random.shuffle(list)
    list = list[0: N1]
    input2 = inputimg[list, :, :, :]
    input2s = inputseg[list, :, :, :]
    #dim = np.asarray(input2.shape, dtype=int)  # dim is 4x1, NxHxWxC, N= number of patches
    #input4 = np.zeros(dim, dtype=np.float32)
    #input4s = np.zeros(dim, dtype=np.float32)
    count = N+N1
    for i in range(0, N1):
        for c in range(0, dim[3]):
            outputimg[count + i, :, :, c] = np.fliplr(input2[i, :, :, c])
            outputimg[count+ i, :, :, c] = np.fliplr(input2s[i, :, :, c])
    #outputimg = np.concatenate((outputimg, input3), axis=0)
    #outputseg = np.concatenate((outputseg, input3s), axis=0)

    # 90 degree counterclockwise rotation
    print('90 degree rotation')
    list = np.asarray(range(0, N), dtype=int)
    np.random.seed(int(datetime.now().timestamp()))
    np.random.shuffle(list)
    list = list[0: N1]
    input2 = inputimg[list, :, :, :]
    input2s = inputseg[list, :, :, :]
    #dim = np.asarray(input2.shape, dtype=int)  # dim is 4x1, NxHxWxC, N= number of patches
    #input3 = np.zeros(dim, dtype=np.float32)
    #input3s = np.zeros(dim, dtype=np.float32)
    count = N + 2*N1
    for i in range(0, N1):
        for c in range(0, dim[3]):
            outputimg[count + i, :, :, c] = np.rot90(input2[i, :, :, c])
            outputseg[count + i, :, :, c] = np.rot90(input2s[i, :, :, c])
    #outputimg = np.concatenate((outputimg, input3), axis=0)
    #outputseg = np.concatenate((outputseg, input3s), axis=0)

    # random rotation between -45 and 45 degrees
    print('Random rotation between -45 and +45 degrees')
    list = np.asarray(range(0, N), dtype=int)
    np.random.seed(int(datetime.now().timestamp()))
    np.random.shuffle(list)
    list = list[0: N1]
    input2 = inputimg[list, :, :, :]
    input2s = inputseg[list, :, :, :]
    #dim = np.asarray(input2.shape, dtype=int)  # dim is 4x1, NxHxWxC, N= number of patches
    #input3 = np.zeros(dim, dtype=np.float32)
    #input3s = np.zeros(dim, dtype=np.float32)
    count = N +3*N1
    a = np.random.rand(N1)
    a = 45*(2*a -1)
    for i in range(0, N1):
        for c in range(0, dim[3]):
            outputimg[count + i, :, :, c] = rotate(input2[i, :, :, c],a[i], reshape=False, order=0, mode='reflect')
            outputseg[count + i, :, :, c] = rotate(input2s[i, :, :, c], a[i], reshape=False, order=0, mode='reflect')
    #outputimg = np.concatenate((outputimg, input3), axis=0)
    #outputseg = np.concatenate((outputseg, input3s), axis=0)

    # random zooming
    print('Random zooming between 1 and 1.25')
    list = np.asarray(range(0, N), dtype=int)
    np.random.seed(int(datetime.now().timestamp()))
    np.random.shuffle(list)
    list = list[0: N1]
    input2 = inputimg[list, :, :, :]
    input2s = inputseg[list, :, :, :]
    #dim = np.asarray(input2.shape, dtype=int)  # dim is 4x1, NxHxWxC, N= number of patches
    #input3 = np.zeros(dim, dtype=np.float32)
    #input3s = np.zeros(dim, dtype=np.float32)
    a = np.random.rand(N1)
    a = 1+a/4  # zoom between 1 and 1.25
    count = N+4*N1
    for i in range(0, N1):
        for c in range(0, dim[3]):
            outputimg[count + i, :, :, c] = cv2_clipped_zoom(input2[i, :, :, c], a[i])
            outputseg[count + i, :, :, c] = cv2_clipped_zoom(input2s[i, :, :, c], a[i])
    #outputimg = np.concatenate((outputimg, input3), axis=0)
    #outputseg = np.concatenate((outputseg, input3s), axis=0)

    '''
    # random gamma correction, probably not a good idea because the image intensities are important and
    # nonlinear scaling is bad
    print('Random Gamma between 0.75 and 1.25')
    list = np.asarray(range(0, N), dtype=int)
    np.random.shuffle(list)
    list = list[0: N1]
    input2 = inputimg[list, :, :, :]
    input2s = inputseg[list, :, :, :]
    dim = np.asarray(input2.shape, dtype=int)  # dim is 4x1, NxHxWxC, N= number of patches
    input3 = np.zeros(dim, dtype=np.float32)
    input3s = np.zeros(dim, dtype=np.float32)
    a = np.random.rand(dim[0])
    a = 1+ a/2 - 0.25 # between 0.75 and 1.25
    for i in range(0, dim[0]):
        for c in range(0, dim[3]):
            input3[i, :, :, c] = adjust_gamma(input2[i, :, :, c], a[i])
            input3s[i, :, :, c] = input2s[i, :, :, c]
    outputimg = np.concatenate((outputimg, input3), axis=0)
    outputseg = np.concatenate((outputseg, input3s), axis=0)
    '''
    inputimg = outputimg
    inputseg = outputseg
    return inputimg, inputseg