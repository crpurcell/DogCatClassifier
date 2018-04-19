#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     1_classify_simple.py                                              #
#                                                                             #
# PURPOSE:  Demonstrate a simple CNN to classify cats & dogs using keras API. #
#           Based on an original post from https://blog.keras.io/building-    #
#           powerful-image-classification-models-using-very-little-data.html  #
#           Data available from https://www.kaggle.com/c/dogs-vs-cats/data    #
#                                                                             #
# DATA:     data/train/dogs                                                   #
#           data/train/cats ... training data (1000 images each class)        #
#                                                                             #
#           data/validation/dogs                                              #
#           data/validation/cats ... validation data (400 images each)        #
#                                                                             #
# MODIFIED: 18-Apr-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#
from __future__ import print_function

# Training and validation data directories
dirTrain = "data/train"
dirValid = "data/validation"

# Image dimensions
imgWidth, imgHeight = 150, 150

# Total number of train & validation images from all classes
nTrainImg = 2000
nValidImg = 800

# Number of training epochs
epochs = 40

# Batch size
batchSize = 32

#-----------------------------------------------------------------------------#

import sys
import pickle
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras


#-----------------------------------------------------------------------------#
def main():

    # Determine what shape the back-end (tensorflow) requires
    if K.image_data_format() == 'channels_first':
        inShape = (3, imgWidth, imgHeight)
    else:
        inShape = (imgWidth, imgHeight, 3)
    
    # Use the Sequential API to define linear stack of layers
    # 1st layer should have a defined input shape
    model = Sequential()
    print("Defining the model ...", end="")
    sys.stdout.flush()
    
    # 1st layer: convolution with a 3x3 kernel, 32 output filters
    #            'rectified linear unit' normalisation
    #            pool to decrease size of output
    model.add(Conv2D(32, (3, 3), input_shape=inShape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # 2nd layer: convolution with a 3x3 kernel, 32 output filters
    #            'rectified linear unit' normalisation
    #            MaxPooling to decrease size of output
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3rd layer: convolution with a 3x3 kernel, 64 output filters
    #            'rectified linear unit' normalisation
    #            MaxPooling to decrease size of output
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output image and connect to 64 neurons
    # Dense = fully connected (every output pix to every neuron)
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))

    # Randomly set 1/2 last outputs to 0
    # Set a sigmoid activation function to allow non-linearity
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile the model and use 'cross-entropy' as a measure of how well
    # the algorithm is performing (i.e., for training)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print("done.")
    sys.stdout.flush()

    #-------------------------------------------------------------------------#
    # Now augment the input data and train the network

    # Augment the training data by doing some random transformations. Helps
    # generalise the network and prevents over-fitting.
    print("Setting augmentation ...", end="")
    sys.stdout.flush()
    trainDataGen = ImageDataGenerator(
        rescale=1./255,          # scale RGB range (1-255) to 0-1
        shear_range=0.2,         # random shear transformation
        zoom_range=0.2,          # random zoom
        #width_shift_range=0.2,   # shift images horizontally 
        #height_shift_range=0.2,  # shift images vertically
        #rotation_range=179,      # randomly rotate
        horizontal_flip=True)    # flip horizontal
    
    # For testing, we want to normalise RGB to 0-1 range, otherwise weights
    # will need to have too high a range for typical learning rate.
    validDataGen = ImageDataGenerator(rescale=1. / 255)
    print("done.")
    sys.stdout.flush()
    
    # Define a training directory iterator set to resizing on the flow.
    # Assumes one sub-directory per class, ordered alpha-numerically and
    # will create an augmented batch of images when iterated.
    print("Defining directory iterators: ")
    sys.stdout.flush()
    trainGen = trainDataGen.flow_from_directory(
        dirTrain,
        target_size=(imgWidth, imgHeight),
        batch_size=batchSize,
        class_mode='binary')

    # Define a validation directory iterator
    validGen = validDataGen.flow_from_directory(
        dirValid,
        target_size=(imgWidth, imgHeight),
        batch_size=batchSize,
        class_mode='binary')   # Two classes
    
    # Finally, train the network
    raw_input("\nReady to start training. Press <RETURN>: ")
    timeStart = time.time()
    history = model.fit_generator(
        trainGen,
        steps_per_epoch=nTrainImg // batchSize,
        epochs=epochs,
        validation_data=validGen,
        validation_steps=nValidImg // batchSize)
    hist = history.history
    timeEnd = time.time()
    runTime_min = (timeEnd-timeStart)/60.0
    print("Training took %.1f minutes." % runTime_min)
    
    # Save the weights in a HDF5 file
    model.save_weights('1_classifier_weights.h5')

    # Save the history in a pickle
    fh = open("1_classifier_history.pkl", 'w')
    pickle.dump(hist, fh)

    # Plot the loss and accuracy history
    mpl.rcParams['font.size'] = 12.0
    fig = plt.figure(figsize=(14., 6.))
    ax1 = fig.add_subplot(1,2,1)
    epoch = range(1, len(hist['loss'])+1)
    ax1.step(epoch, hist['loss'], where="mid", label="Training Loss")
    ax1.step(epoch, hist['val_loss'], where="mid", label="Validation Loss")
    ax1.legend(loc='upper right', shadow=False, fontsize='medium')
    ax1.set_title("Model Loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax2 = fig.add_subplot(1,2,2)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.step(epoch, hist['acc'], where="mid", label="Training Accuracy")
    ax2.step(epoch, hist['val_acc'], where="mid", label="Validation Accuracy")
    ax2.legend(loc='lower right', shadow=False, fontsize='medium')
    ax2.set_title("Model Accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    
    # Nice formatting
    ax1.tick_params(pad=7)
    for line in ax1.get_xticklines() + ax1.get_yticklines():
        line.set_markeredgewidth(1)
        ax2.tick_params(pad=7)
    for line in ax2.get_xticklines() + ax2.get_yticklines():
        line.set_markeredgewidth(1)
    plt.tight_layout()
    fig.savefig("1_classifier_accuracy.pdf", format='pdf' )    
    fig.show()
    raw_input("Press <RET> to exit ...")
    

#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
