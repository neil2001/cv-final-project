# RoomMesh: Semantic Reconstruction of Indoor Scenes using 2D-3D Learning

This repository contains the submodules and Jupyter Notebooks that I used for my Computer Vision final project. This project seeks to perform semantic reconstruction of indoor scenes using joint 2D-3D learning. It is inspired by [TerrainMesh](https://acsweb.ucsd.edu/~qif007/TerrainMesh/index.html) and makes use of [ESANet](https://github.com/TUI-NICR/ESANet) for semantic segmentation and labelling of indoor images.

To run indoor experiments, I make use of the NYUV2 dataset with 40 classes, which can be accessed [here](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).

My project additionally examines [Kimera](https://arxiv.org/abs/1910.02490), an open source library for performing metric-semantic localization and mapping. Details surrounding my interaction with this project are shown below.

Slides for my project are linked [here](https://docs.google.com/presentation/d/1tdAW4aifPOWevr2-8ch69ibqcakpTdT6EhswoN229cU/edit?usp=sharing).

## RoomMesh Repository
This repository contains the following notebooks that encompass the experiments I ran:
- Experiments-TerrainMesh: this notebook comprises some of the experiments performed in the TerrainMesh paper. I make use of the WHU dataset, which can be accessed [here](https://github.com/FengQiaojun/TerrainMesh_Data).
- Experiments-Indoor-Initial: this notebook contains initial experiments ran using TerrainMesh on indoor scenes.
- Experiments-RoomMesh-Dev: this notebook contains experiments I ran in order to develop the mesh smoothing and remapping technique, as well as integrate ESANet into the RoomMesh pipeline.
- Experiments-RoomMesh: this notebook contains experiments I used to evaluate the geometric and semantic accuracy of RoomMesh agains the NYUv2 dataset.

## Kimera Installation
The following steps detail how I installed and ran experiments using Kimera:
1. Create a virtual Ubuntu 20.04 instance on my Intel Macbook Air using UTM
2. Install ROS Noetic
3. Install [Kimera-VIO](https://github.com/MIT-SPARK/Kimera-VIO-ROS)
4. Install [Kimera-Semantics](https://github.com/MIT-SPARK/Kimera-Semantics/tree/master)
5. Follow [this tutorial](https://www.youtube.com/watch?v=Zjevg5wQTdI&t=947s) to run experiments