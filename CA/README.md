# Thesis
## Requirements

- **Cuda & Cudnn & Python & Pytorch**

    This project is tested with CUDA 8.0/9.0, Python 3, Pytorch 0.4/1.0/1.1, NVIDIA Tesla V100/TITANX GPU. And almost all the packages we use are covered by Anaconda.

    Please install proper CUDA and CUDNN version, and then install Anaconda3 and Pytorch.

- **My settings**

  ```shell
	torch                              1.1.0
	torchfile                          0.1.0
	torchvision                        0.3.0
	numpy                              1.14.3
	numpydoc                           0.8.0
	numba                              0.38.0
	visdom                             0.1.8.9
	opencv-python                      4.1.0.25
	easydict                           1.9
	Shapely                            1.6.4.post2
  ```

## Data preparation

Download and unzip the full [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) detection dataset to the folder **/path/to/kitti/** (left color, left 3 temporally preceding, camera calibration and labels). Then place a softlink (or the actual data) in **data/kitti/**. There are two widely used training/validation set splits for the KITTI dataset. Here we only show the setting of **split1**, you can set **split2** accordingly.

  ```shell
ln -s /path/to/kitti data/kitti
ln -s /path/to/kitti/testing data/kitti_split1/testing
  ```

Our method uses [DORN](https://github.com/hufu6371/DORN) (or other monocular depth models) to extract depth maps for all images. You can download and unzip the depth maps extracted by DORN [here](https://drive.google.com/open?id=1lSJpQ8GUCxRNtWxo0lduYAbWkkXQa2cb) and put them (or softlink) to the folder **data/kitti/depth_2/**. (You can also change the path in the scripts **setup_depth.py**)

Then use the following scripts to extract the data splits, which use softlinks to the above directory for efficient storage.


  ```shell
python data/kitti_split1/setup_split.py
python data/kitti_split1/setup_depth.py
  ```

Next, build the KITTI devkit eval for split1.

```shell
sh data/kitti_split1/devkit/cpp/build.sh
```

Lastly, build the nms modules

```shell
cd lib/nms
make
```

## Best Model Download Link
https://drive.google.com/file/d/1-GM1TgHGCeIkRmQ4EoiLU3qXRIt8Cc2y/view?usp=sharing

## Training
```shell
python scripts/train_rpn_3d.py --config=config
```

## Validation
```shell
./data/kitti_split1/devkit/cpp/evaluate_object /your/results/data
```

## Testing
```shell
python scripts/test.py --config=config
```

## Generate Synthetic Data
```shell
python warping.py
```

## Visualization
https://github.com/kuixu/kitti_object_vis
