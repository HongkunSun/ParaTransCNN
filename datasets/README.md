# Data Preparing
## Synapse Data
1. Access to the synapse multi-organ dataset:
   1. Sign up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/) and download the dataset. Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keeping the 3D volume in h5 format for testing cases.
2. The directory structure of the whole project is as follows:

```bash
└── data
    └──Synapse
        ├── test_vol
        │   ├── case0001.npy.h5
        │   └── *.npy.h5
        └── train
            ├── case0005_slice000.npz
            └── *.npz
```

## Aortic Vessel Tree(AVT) Data
1. Access to the aortic vessel tree dataset:
   1. Use the [AVT data website](https://figshare.com/articles/dataset/Aortic_Vessel_Tree_AVT_CTA_Datasets_and_Segmentations/14806362) and download the dataset(KiTS,Rider,Dongyang). First, the data is resampled to 1mm x 1mm x Z, converte them to numpy and clip to [-190, 1668], and normalize to [0,1] . Select 2D slices with masks from the 3D volume as the training cases while keeping the 3D volume in h5 format for testing cases.
2. The directory structure of the whole project is as follows:
```bash
└── data
    └──AVT
        ├── test_vol
        │   ├── R15.h5
        │   ├── K19.h5
        │   └── *.h5
        └── train
            ├── K17_slice_117.npz
            ├── R5 (AD)_slice_407.npz
            └── *.npz
```
