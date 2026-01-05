# AB-MRUNet

We recommend using a conda environment over a virtual environment. This repository is base on the official PyTorch code for the FR-UNet( ['Full-Resolution Network 
and Dual-Threshold Iteration for Retinal Vessel and Coronary Angiograph Segmentation'](https://ieeexplore.ieee.org/abstract/document/9815506).)

## Setup repository and environment

```
# clone our repository
git clone git@github.com:Brainsmatics-Opticalimaging/AB-MRUNet.git
cd AB-MRUNet

# Create a Conda Environment
conda create --name AB-MRUNet

# Activate the Conda Environment
conda activate AB-MRUNet

# Install Required Packages
pip install -r requirements.txt
```

## Datasets processing

Choose a path to create a folder with the dataset name and download datasets [DRIVE](https://www.dropbox.com/sh/z4hbbzqai0ilqht/AAARqnQhjq3wQcSVFNR__6xNa?dl=0),[CHASEDB1](https://blogs.kingston.ac.uk/retinal/chasedb1/),[STARE](https://cecas.clemson.edu/~ahoover/stare/probing/index.html),[CHUAC](https://figshare.com/s/4d24cf3d14bc901a94bf), and [DCA1](http://personal.cimat.mx:8181/~ivan.cruz/DB_Angiograms.html). Type this in terminal to run the data_process.py file

```
python data_process.py -dp DATASET_PATH -dn DATASET_NAME
```

## Training

Type this in terminal to run the train.py file

```
python train.py -dp DATASET_PATH
```

## Test

Type this in terminal to run the test.py file

```
python test.py -dp DATASET_PATH -wp WEIGHT_FILE_PATH
```

We have prepared the pre-trained models for both datasets in the folder 'pretrained_weights'. To replicate the results in the paper, directly run the following commands

```
python test.py -dp DATASET_PATH -wp pretrained_weights/DATASET_NAME
```

## Inference

Type this in terminal to run the prediction.py file

```
python prediction.py
```

## Pretrained Model/weights and Dataset download（Coming soon）:

Please download the model folder from the following Baiduyun Drive link: （coming soon）



