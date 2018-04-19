#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     classify_finetune.py                                              #
#                                                                             #
# PURPOSE:  Demonstrate how to fine-tune the last convolutional layer and     #
#           final dense layer on the VGG network for the cats-vs-dogs         #
#           problem. We make use of the weights learned for the final dense   #
#           layer in the frozen bottleneck script.                            #
#                                                                             #
#           Original post from https://blog.keras.io/building-                #
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
nTrainImg = 1984
nValidImg = 800

# Number of training epochs
epochs = 40

# Batch size
batchSize = 64

# File containing pre-trained weights for dense layer (top model).
# Requires a good starting guess and a slow learning rate to converge.
topModelWeightFile = "2_classifier_weights.h5"


#-----------------------------------------------------------------------------#

import sys
import pickle
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input
from keras import optimizers
from keras import applications

#-----------------------------------------------------------------------------#
def main():

    # Load the pre-trained VGG16 convolution layers and weights
    inputTensor = Input(shape=(imgWidth, imgWidth, 3))
    VGG16 = applications.VGG16(include_top=False, weights='imagenet',
                               input_tensor=inputTensor)
    model = Sequential()
    for layer in VGG16.layers:
        model.add(layer)
    
    # Build a simple fully-connected model do the classification.
    print("Defining the sequential model and adding it to VGG16 ...", end="")
    sys.stdout.flush()
    topModel = Sequential()
    topModel.add(Flatten(input_shape=model.output_shape[1:]))
    topModel.add(Dense(256, activation='relu'))
    topModel.add(Dropout(0.5))
    topModel.add(Dense(1, activation='sigmoid'))

    # Load the initial weights of the top model
    topModel.load_weights(topModelWeightFile)
    
    # add the fully connected layer on top of the convolutional base
    model.add(topModel)
    
    # Freeze the the first 15 layers (up to the last conv block)
    for layer in model.layers[:15]:
        layer.trainable = False

    # Compile the model and use 'cross-entropy' as performance measure
    # Use a SGD/momentum optimizer (not an adaptive optimiser, e.g., RMSProp)
    # Use a slow learning rate to avoid diverging from pre-trained weights
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    print("done.")
    sys.stdout.flush()

    #-------------------------------------------------------------------------#
    # Now augment the input data and train the network

    # Augment the training data
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

    # For testing, we want to normalise RGB to 0-1 range
    validDataGen = ImageDataGenerator(rescale=1. / 255)
    print("done.")
    sys.stdout.flush()
    
    # Define a training directory iterator
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
        class_mode='binary')

    # Finally, train the network
    # TODO: Why is this different than before?
    raw_input("\nReady to start training. Press <RETURN>: ")
    timeStart = time.time()
#    history = model.fit_generator(
#        trainGen,
#        samples_per_epoch=nTrainImg,
#        epochs=epochs,
#        validation_data=validGen,
#        nb_val_samples=nValidImg)
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
    model.save_weights('3_classifier_weights.h5')

    # Save the history in a pickle
    fh = open("3_classifier_history.pkl", 'w')
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
    fig.savefig("3_classifier_accuracy.pdf", format='pdf')
    fig.show()
    raw_input("Press <RET> to exit ...")

    
#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
