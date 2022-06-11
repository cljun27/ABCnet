# ABCnet
Adversarial Bias Correction Network for Infant Brain MR Images.

## Architecture
![ABCnet](https://github.com/cljun27/ABCnet/blob/main/images/arc_j3.png).

## How to use it
### Prerequisites
- Linux
- NVIDIA GPU with at least 8Gb memory
- CUDA CuDNN

### Python Dependencies
- python (3.6)
- pytorch (0.4.1+)
- torchvision (0.2.1+)
- tensorboardx (1.6+)

### Data preparation
Resample and convert the input files into npy format with size of 256x256x256.

### Train
Modify the [train.py](https://github.com/cljun27/ABCnet/blob/main/train.py) file to match the training data in your own path. Then, run:
```
python train.py --dataroot /training_data_folder --name project_name --model pix2pix3d --direction AtoB --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0
```

### Test

```
python test.py --dataroot /data_folder --name project_name --model pix2pix3d --direction AtoB --dataset_mode unaligned --input_nc 1 --output_nc 1 --gpu_ids 0
```

### Citations
If you use this code for your research, please cite as:

Chen, Liangjun, et al. "ABCnet: Adversarial bias correction network for infant brain MR images." Medical Image Analysis 72 (2021): 102133.
