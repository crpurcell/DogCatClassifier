# CNNs to classify sparse samples of images

This repository contains a set of scripts that demonstrate how to
construct a simple Convolutional Neural Network to classify images of
cats and dogs. The python code uses the keras API and is based on an
original 2016 [blog
post](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
by Francois Chollet. The data for this example is available from [kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)


## Usage

Create a directory tree as follows
```
├── data
    ├── train
    │   ├── cats
    │   └── dogs
    └── validation
        ├── cats
        └── dogs
```

and populate the ```train``` with 1000 images of cats and 1000 images
of dogs. Put 400 images in each class within the validation directory.

Make the python scripts executable using ```chmod u+x *.py``` and
execute in turn.

Requires a working installation of keras, in addition to the python
modules ```numpy, matplotlib, pillow, h5py```.