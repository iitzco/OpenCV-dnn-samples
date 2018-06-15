# OpenCV dnn samples

Here you can find 3 different examples (Tensorflow, Caffe and Torch) on how to use the `dnn` package from OpenCV. The idea is to understand how the package can be used to make inferences on any trained model.

For each model used, you need to understand the meaning of their output values.

> **Attention**: this code runs under Python3

## Why is this cool?

There is no need to install any Deep Learning framework! With just OpenCV you can make inferences on trained graphs from these common frameworks: Tensorflow, Caffe and Torch.

## How to use?

1. Clone the repo.
2. *Optional* Create a virtual environment (for example, using `virtualenv`) to keep dependencies isolated.
3. Run `pip install -r requirements.txt`
4. Run any of the following samples:

#### Tensorflow

You can run an Object Detection model based on SSD+MobileNet trained on COCO dataset.

```bash
$ python main_tensorflow.py ./images/people.jpg
```

The used model was extracted from the official Tensorflow Object Detection API Zoo [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

#### Caffe

You can run an Image Classification Detection model trained on ImageNet dataset.

```bash
$ python main_caffe.py ./images/eagle.png
```

This model was extracted from the code that [this](https://www.pyimagesearch.com/2017/08/21/deep-learning-with-opencv/) excellent post provides.

#### Torch

You can run an Image Classification Detection model trained on ImageNet dataset.

```bash
$ python main_torch.py ./images/eagle.png
```

The model needs to be downloaded because it's not included in this repo. Go to `./torch/` to get the link to the official download URL. The model was extracted from [here](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained).

## Acknowledge

* Refer to [this blogpost](https://habr.com/company/intel/blog/333612/) written by the author of the `dnn` package.
* More information here: https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV
