# bacteria_segmentation
 pretrained and instructions for annotating and training segmentation of bacterial fluorescent images

## Installation

Create environment:
```sh
conda create --name napari-env
conda activate napari-env
conda install -c conda-forge napari   
conda install opencv
```

Install Tensorflow for macOS M1/M2:
```sh
pip install tensorflow-macos
pip install tensorflow-metal
```

Install stardist for cell segmentation:
```sh
pip install gputools
pip install stardist
pip install csbdeep

pip install splinedist
```

## How to use

### Data augmentation

```sh
python augment_image_data.py
```

### Perform training

```sh
python training.py
```
Open up  tensorboard to follow the results:
```sh
tensorboard --logdir=.
```

### Prediction

```sh
python prediction.py './examples/' 
```

Where `examples` is the folder that contains `.tif` files for analysis.
Format has to be 16-bit monochrome TIFF.

The output will be produced in the active wokring directory as the original file names with prefix `output_`.

- `output_*.tif` labels, 16-bit TIFF each cell has a unique unint16 value.
- `output_*.zip` ImageJ ROI zip file, can be opened in ImageJ/Fiji ROI Manager.