# 3dr2n2-Deep-Learning-Course-project


## Introduction

This repository contains the source codes for the paper Choy et al., 3D-R2N2:
A Unified Approach for Single and Multi-view 3D Object Reconstruction, ECCV 2016.
Given one or multiple views of an object, the network generates voxelized
( a voxel is the 3D equivalent of a pixel) reconstruction of the object in 3D.\\

This repository is a part of Deep Learning Course Project which contains two phases.
The first phase is to re-implement 3dr2n2 from scratch using tensorflow. The second phase
is to improve the performance of 3d reconstructionddd.\\


## Datasets

We used ShapeNet models to generate rendered images and voxelized models which are available below:
To download, use wget ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz.

* [ShapeNet rendered images ](ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz)
* [ShapeNet voxelized models ](ftp://cs.stanford.edu/cs/cvgl/ShapeNetVox32.tgz)


## Training the network

* Download datasets and place them in a folder data
* Change dir to data_loader and run `python save_files.py`
* Change dir to mains and run `python train.py`



