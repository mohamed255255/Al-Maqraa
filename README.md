# **QuartzNet Model**

The QuartzNet is a better variant of Jasper with a key difference that it uses time-channel separable 1D convolutions. This allows it to dramatically reduce the number of weights while keeping similar accuracy.

A Jasper/QuartzNet models look like this (QuartzNet model is pictured):

![QuartzNet Model](./quartez)

## **Requirements**

- Python 3.10 or above
- PyTorch 1.13.1 or above
- NVIDIA GPU (if you intend to do model training)

## **Use Anaconda to Avoid Package Conflicts**

```sh
conda create --name condaEnv python==3.10.12
conda activate condaEnv
