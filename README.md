<h1 align="center">Terabyte scale image denoising using multiple nodes & multiple GPUs</h1>

<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains Python based image denoising scripts using common CNN
architectures like Unet and its variations, RCAN, Inception etc. The primary focus
is to train a model from a limited training dataset (e.g. 3-5) and apply it to
very large multi-Terabyte 3D images (e.g. 100k x 100k pixels with thousands of slices)
using multiple GPUs on multiple nodes. A simple GUI is also provided to train/predict
small images.

## Overview
