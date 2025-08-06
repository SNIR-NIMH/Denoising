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
from tkinter import filedialog, LabelFrame
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
root.title('Applying denoised trained model')
# setting the windows size
root.geometry("800x200")  # Width x height
root.resizable(False,False)

def getInputFolderPath():
    folder_selected = filedialog.askdirectory()
    inputpath.set(folder_selected)

def getOutputFolderPath():
    folder_selected = filedialog.askdirectory()
    outputpath.set(folder_selected)

def getModelPath():
    folder_selected = filedialog.askopenfilename(filetypes=[('HDF5 files',"*.h5")])
    modelpath.set(folder_selected)

def submit():
    imgdir = inputpath.get()
    outdir = outputpath.get()
    model = modelpath.get()
    p1 = psize1.get()
    p2 = psize2.get()
    p3 = psize3.get()

    c_h = chunk_h.get()
    c_w = chunk_w.get()
    path = os.path.dirname(sys.argv[0])
    path = os.path.abspath(path)
    path = os.path.join(path, 'Denoise_Test.py ')
    # Use --atlasdir="path" --> The double quote and equal-to ensures the space in the path is respected
    # Using --atlasdir path or --atlasdir "path" does not work if there are spaces in  path, only arg equalto quote path unquote works
    if p3==1:
        cmd = 'python ' + path + ' --im="' + imgdir + '" --psize ' + str(p1) + ' ' + str(p2) + ' --modalities unk --network unet --o="' + outdir  \
           + '" --gpu 0 ' + ' --chunks ' + str(c_h) + ' ' + str(c_w) + ' --model="' + str(model) + '"'
    else:
        cmd = 'python ' + path + ' --im="' + imgdir + '" --psize ' + str(p1) + ' ' + str(p2) + ' ' + str(p3) +' --modalities unk --network unet --o="' + \
              outdir + '" --gpu 0 ' + ' --chunks ' + str(c_h) + ' ' + str(c_w) + ' --model="' + str(model) + '"'
    print(cmd)
    os.system(cmd)


    root.destroy()


if __name__ == "__main__":


    # declaring string variable
    # for storing name and password
    #atlasdir_var = tk.StringVar()
    inputpath = tk.StringVar()
    outputpath = tk.StringVar()
    modelpath = tk.StringVar()

    psize1 = tk.IntVar()
    psize2 = tk.IntVar()
    psize3 = tk.IntVar()
    chunk_h = tk.IntVar()
    chunk_w = tk.IntVar()


    # One frame for the folder inputs
    frame1 = LabelFrame(root)
    a = tk.Label(frame1, text="Input folder containing 2D tifs", padx=10)
    a.grid(row=1, column=1)
    E = tk.Entry(frame1, textvariable=inputpath, width=40)
    E.grid(row=1, column=2, ipadx=60)
    btnFind = ttk.Button(frame1, text="Browse Folder", command=getInputFolderPath)
    btnFind.grid(row=1, column=3)

    a = tk.Label(frame1, text="Output folder where 2D slices will be written", padx=10)
    a.grid(row=2, column=1)
    E = tk.Entry(frame1, textvariable=outputpath, width=40)
    E.grid(row=2, column=2, ipadx=60)
    btnFind = ttk.Button(frame1, text="Browse Folder", command=getOutputFolderPath)
    btnFind.grid(row=2, column=3)

    a = tk.Label(frame1, text="Trained model in h5 format", padx=10)
    a.grid(row=3, column=1)
    E = tk.Entry(frame1, textvariable=modelpath, width=40)
    E.grid(row=3, column=2, ipadx=60)
    btnFind = ttk.Button(frame1, text="Browse File", command=getModelPath)
    btnFind.grid(row=3, column=3)
    frame1.grid(row=0, column=0, sticky='ew')

    # Second frame for the numeric inputs
    frame2 = LabelFrame(root)
    psize_label = tk.Label(frame2, text='Patch size in HxWxD (same one used for training)')
    chunk_h_label = tk.Label(frame2, text='Number of horizontal chunks')
    chunk_w_label = tk.Label(frame2, text='Number of vertical chunks')


    psize1_entry = tk.Entry(frame2, textvariable=psize1, width=5)
    psize2_entry = tk.Entry(frame2, textvariable=psize2, width=5)
    psize3_entry = tk.Entry(frame2, textvariable=psize3, width=5)
    chunk_h_entry = tk.Entry(frame2, textvariable=chunk_h, width=5)
    chunk_w_entry = tk.Entry(frame2, textvariable=chunk_w, width=5)
    psize1.set(128)
    psize2.set(128)
    psize3.set(1)
    chunk_w.set(2)
    chunk_h.set(2)



    psize_label.grid(row=4, column=1, padx=60)
    chunk_h_label.grid(row=5, column=1, padx=60)
    chunk_w_label.grid(row=6, column=1, padx=60)


    psize1_entry.grid(row=4, column=2, padx=5)
    psize2_entry.grid(row=4, column=3, padx=5)
    psize3_entry.grid(row=4, column=4, padx=5)

    chunk_h_entry.grid(row=5, column=3, padx=5)
    chunk_w_entry.grid(row=6, column=3, padx=5)
    frame2.grid(row=1, column=0, sticky='ew')

    # Different frame for Run
    frame3 = LabelFrame(root,labelanchor='n')
    sub_btn = tk.Button(frame3, text='Run', command=submit)
    sub_btn.grid(row=0,column=0)
    frame3.grid(row=2, column=0, padx=20)

    # performing an infinite loop
    # for the window to display
    root.mainloop()
