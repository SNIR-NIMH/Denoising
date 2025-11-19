import os
from glob import  glob
import sys
from PIL import Image
from tqdm import tqdm
import argparse
from skimage.io import imread, imsave
import numpy as np
#from pytiff import Tiff
#import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, LabelFrame, Button
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
x = tf.test.is_gpu_available()
#x = tf.config.list_logical_devices()
if x == False:
    sys.exit('ERROR: GPU is not available. Denoising training will not work')

warnings.filterwarnings("ignore")

path = os.path.dirname(sys.argv[0])
path = os.path.abspath(path)
#print('Appending {}'.format(path))
sys.path.append(path)



# ======================================================================
root = tk.Tk()
root.title('Denoise Training')
# setting the windows size
root.geometry("800x200")  # Width x height
root.resizable(False,False)


def getFolderPath():
    folder_selected = filedialog.askdirectory()
    folderPath.set(folder_selected)
    #dir = os.path.dirname(folder_selected)
    #base = os.path.basename(folder_selected)
    #base = base + '_downsampled.tif'
    #name = os.path.join(dir,base)
    #output_var.set(name)


def submit():
    atlasdir = folderPath.get()
    p1 = psize1.get()
    p2 = psize2.get()
    p3 = psize3.get()

    n = natlas.get()
    mx = maxpatch.get()
    bf = basefilters.get()
    bs = batchsize.get()
    e = epoch.get()
    g = gpuid.get()
    path = os.path.dirname(sys.argv[0])
    path = os.path.abspath(path)
    path = os.path.join(path, 'Synthesis_Train.py ')
    # Use --atlasdir="path" --> The double quote and equal-to ensures the space in the path is respected
    # Using --atlasdir path or --atlasdir "path" does not work if there are spaces in  path, only arg equalto quote path unquote works
    if p3==1:
        cmd = 'python ' + path + ' --atlasdir="' + atlasdir + '" --natlas ' + str(n) + ' --psize ' + str(p1) + ' ' + str(p2) +  \
          ' --modalities unk,unk --model unet --o="' + atlasdir + '" --gpu ' + g + ' --maxpatch ' + str(mx) + ' --basefilters ' + str(bf) + \
             ' --batchsize ' + str(bs) + ' --epoch ' + str(e) + ' --loss mae '
    else:
        cmd = 'python ' + path + ' --atlasdir="' + atlasdir + '" --natlas ' + str(n) + ' --psize ' + str(p1) + ' ' + str(p2) + ' ' + str(p3) + \
              ' --modalities unk,unk --model unet --o "' + atlasdir + '" --gpu ' + g + ' --maxpatch ' + str(mx) + ' --basefilters ' + str(bf) + \
              ' --batchsize ' + str(bs) + ' --epoch ' + str(e) + ' --loss mae '

    print(cmd)
    os.system(cmd)


    root.destroy()


if __name__ == "__main__":


    # declaring string variable
    # for storing name and password
    #atlasdir_var = tk.StringVar()
    natlas = tk.IntVar()
    psize1 = tk.IntVar()
    psize2 = tk.IntVar()
    psize3 = tk.IntVar()
    maxpatch = tk.IntVar()
    basefilters = tk.IntVar()
    batchsize  = tk.IntVar()
    epoch = tk.IntVar()
    gpuid = tk.StringVar()

    # One frame for atlas folder because it has 3 columns
    frame1 = LabelFrame(root)

    folderPath = tk.StringVar()
    a = tk.Label(frame1, text="Atlas folder", padx=25)
    a.grid(row=1, column=1)
    E = tk.Entry(frame1, textvariable=folderPath, width=65)
    E.grid(row=1, column=2, ipadx=60)
    btnFind = ttk.Button(frame1, text="Browse Folder", command=getFolderPath)
    btnFind.grid(row=1, column=3)
    frame1.grid(row=0, column=0, sticky='ew')

    # A different frame for the inputs
    frame2 = LabelFrame(root)

    natlas_label = tk.Label(frame2, text='Number of atlas images')
    psize_label = tk.Label(frame2, text='Patch size in HxWxD (must be multiple of 16)')
    maxpatch_label = tk.Label(frame2, text='Maximum number of patches')
    basefilters_label = tk.Label(frame2, text='Number of base filters')
    batchsize_label = tk.Label(frame2, text='Batch size')
    epoch_label = tk.Label(frame2, text='Number of training epochs')
    gpu_label = tk.Label(frame2, text='GPU IDs to use (comma separated string, starting from 0)')

    natlas_entry = tk.Entry(frame2, textvariable=natlas, width=5)
    psize1_entry = tk.Entry(frame2, textvariable=psize1, width=5)
    psize2_entry = tk.Entry(frame2, textvariable=psize2, width=5)
    psize3_entry = tk.Entry(frame2, textvariable=psize3, width=5)
    maxpatch_entry = tk.Entry(frame2, textvariable=maxpatch,  width=10)
    basefilters_entry = tk.Entry(frame2, textvariable=basefilters,  width=5)
    batchsize_entry = tk.Entry(frame2, textvariable=batchsize,  width=5)
    epoch_entry = tk.Entry(frame2, textvariable=epoch,  width=5)
    gpu_entry = tk.Entry(frame2, textvariable=gpuid, width=10)
    psize1.set(128)
    psize2.set(128)
    psize3.set(1)
    maxpatch.set(50000)
    basefilters.set(32)
    batchsize.set(32)
    epoch.set(50)
    gpuid.set(0)

    #c = ttk.Button(root, text="find", command=doStuff)
    #c.grid(row=1, column=4)



    natlas_label.grid(row=2,column=1, padx=60)
    psize_label.grid(row=3, column=1, padx=60)
    maxpatch_label.grid(row=4, column=1, padx=60)
    basefilters_label.grid(row=5, column=1, padx=60)
    batchsize_label.grid(row=6, column=1, padx=60)
    epoch_label.grid(row=7, column=1, padx=60)
    gpu_label.grid(row=8, column=1, padx=60)

    natlas_entry.grid(row=2, column=3, padx=1)
    psize1_entry.grid(row=3, column=2, padx=1)
    psize2_entry.grid(row=3, column=3, padx=1)
    psize3_entry.grid(row=3, column=4, padx=1)
    maxpatch_entry.grid(row=4, column=3, padx=1)
    basefilters_entry.grid(row=5, column=3, padx=1)
    batchsize_entry.grid(row=6, column=3, padx=1)
    epoch_entry.grid(row=7, column=3, padx=1)
    gpu_entry.grid(row=8, column=3, padx=1)

    frame2.grid(row=1, column=0, sticky='ew')

    # Create a separate frame for the Run button
    frame3 = LabelFrame(root, labelanchor='n')
    sub_btn = tk.Button(frame3, text='Run', command=submit)
    sub_btn.grid(row=0, column=0)
    frame3.grid(row=2, column=0, padx=20)

    # performing an infinite loop
    # for the window to display
    root.mainloop()
    

