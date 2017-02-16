# Implementation of VAE, GAN and VAE-GAN in Tensorflow

The implementation is based on the following papers:
* VAE (https://arxiv.org/abs/1312.6114)
* GAN (https://arxiv.org/abs/1406.2661)
* VAE-GAN (https://arxiv.org/abs/1512.09300)

## GAN

This generative adversarial network consists of a generator and discriminator network both of which are multiple 2D convolutions and inverse 2D convolutions respectively.

## VAE

The variational autoencoder uses the same 2D convolutions for encoding and decoding as the GAN.

## VAE-GAN

A bit more complicated.

## Dependencies

* Tensorflow
* Numpy
* Scipy
* Matplotlib

## Usage

TODO: support celeb dataset

### Vanilla GAN
Run
```
python3 training.py --model=GAN
```
in your console.
During training sample images are continously generated.

### Vanilla VAE
Run
```
python3 training.py --model=VAE
```
in your console.

### VAE-GAN
Run
```
python3 training.py --model=VAE-GAN
```
in your console.
