#!/usr/bin/env python
#=============================================================================#
#                                                                             #
# NAME:     classify_pretrained.py                                            #
#                                                                             #
# PURPOSE:  Demonstrate how to load pre-trained weights from a VGG convNet    #
#           and use them to bootsttrap the dogs-vs-cats IDing problem.        #
#           In this version the VGG model is loaded and the 5 convolutional   #
#           layers are attached to a new dense layer.                         #
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
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
    

#-----------------------------------------------------------------------------#
def main():

    # Run the data through the VGG16 network
    #save_VGG_bottleneck()

    # Train a simple sequential model on top of VGG
    train_top_model()
    
    
#-----------------------------------------------------------------------------#
def save_VGG_bottleneck():
    """
    Load the convolutional layers from a pre-trained VGG network and run the 
    training & validation data through it. Save the features at the top of 
    the convolutional layers (pre-dense layers).
    """
    
    # Normalise the input data scale 0 - 1
    # Can't use augmentation for frozen VGG - why?
    print("Setting normalisation of data ...", end="")
    sys.stdout.flush()
    dataGen = ImageDataGenerator(rescale=1. / 255)
    print("done.")
    sys.stdout.flush()

    # Load the VGG16 architecture (part of keras as an 'application')
    # Pre-trained weights are downloaded from the internet.
    # The 3 Dense layers at the top of the network are omitted, so only
    # the convolutional layers are defined. This means the output will
    # consist of the Block 5: 512 output filters.
    model = applications.VGG16(include_top=False, weights='imagenet')
    
    # Define a training directory iterator. Data will be in order, so 1st
    # 1000 images will be cats, then 1000 dogs.
    print("Defining directory iterators: ")
    sys.stdout.flush()
    trainGen = dataGen.flow_from_directory(
        dirTrain,
        target_size=(imgWidth, imgHeight),
        batch_size=batchSize,
        class_mode=None,     # Don't read labels
        shuffle=False)       # Images supplied will be ordered by class

    # Define a validation directory iterator
    sys.stdout.flush()
    validGen = dataGen.flow_from_directory(
        dirValid,
        target_size=(imgWidth, imgHeight),
        batch_size=batchSize,
        class_mode=None,     # Don't read labels
        shuffle=False)       # Images supplied will be ordered by class
    
    # The predict_generator returns the outputs of the pre-trained model
    # given a generator that serves batches of images
    print("Running VGG on training data ...")
    timeStart = time.time()
    bottleneckFeaturesTrain = model.predict_generator(
        trainGen, nTrainImg // batchSize)
    timeEnd = time.time()
    runTime_min = (timeEnd-timeStart)/60.0
    print("Run took %.1f minutes." % runTime_min)
    
    # Save as a binary numpy array
    print("Saving bottleneck features of type %s" %
          type(bottleneckFeaturesTrain))
    np.save(open('2_featuresTrain.npy', 'w'), bottleneckFeaturesTrain)

    # Run VGG on the validation data
    print("Running VGG on validation data ...")
    timeStart = time.time()
    bottleneckFeaturesValid = model.predict_generator(
        validGen, nValidImg // batchSize)
    timeEnd = time.time()
    runTime_min = (timeEnd-timeStart)/60.0
    print("Run took %.1f minutes." % runTime_min)
    
    # Save as a binary numpy array
    print("Saving bottleneck features of type %s" %
          type(bottleneckFeaturesValid))
    np.save(open('2_featuresValid.npy', 'w'), bottleneckFeaturesValid)

    
#-----------------------------------------------------------------------------#
def train_top_model():
    """
    Re-load the trained bottleneck features and build a small fully connected
    network on top of them.
    """

    # Load the trained bottleneck features and assign binary labels
    trainData = np.load(open('2_featuresTrain.npy'))
    nTrainData = trainData.shape[0]
    trainLabels = np.array([0] * (nTrainData / 2) + [1] * (nTrainData / 2))

    # Load the validation bottleneck features and assign binary labels
    validData = np.load(open('2_featuresValid.npy'))
    nValidData = validData.shape[0]
    validLabels = np.array([0] * (nValidData / 2) + [1] * (nValidData / 2))

    # Feedback
    print("Train Shape: %s" % str(trainData.shape))
    print("Valid Shape: %s" % str(validData.shape))
    print("Input Shape: %s" % str(trainData.shape[1:]))
    raw_input("Press <RETURN> ...")
    
    # Build a simple fully-connected model do the classification
    model = Sequential()
    model.add(Flatten(input_shape=trainData.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model and use 'cross-entropy' as performance measure
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # Train the network
    print("Training the FC-network ...")
    sys.stdout.flush()
    timeStart = time.time()
    history = model.fit(trainData,
                        trainLabels,
                        epochs=epochs,
                        batch_size=batchSize,
                        validation_data=(validData, validLabels))
    hist = history.history
    timeEnd = time.time()
    runTime_min = (timeEnd-timeStart)/60.0
    print("Training took %.1f minutes." % runTime_min)
    
    # Save the weights in a HDF5 file
    model.save_weights('2_classifier_weights.h5')

    # Save the history in a pickle
    fh = open("2_classifier_history.pkl", 'w')
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
    fig.savefig("2_classifier_accuracy.pdf", format='pdf')
    fig.show()
    raw_input("Press <RET> to exit ...")
    
    
#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
