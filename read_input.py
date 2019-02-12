import tensorflow as tf
from function import *
import numpy as np
import random
import logging
logging.basicConfig(filename='shrink_2.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def shrink_data():
    image = np.load("./X_test.npy", mmap_mode="r")
    image = image[:,:,:,:,:,0:3]
    logging.warning("image data: \n")
    logging.warning(image[0])
    shape = np.shape(image)
    logging.warning("shape data: \n")
    logging.warning(shape)
    package = np.shape(image)[0]
    logging.warning("package data: \n")
    logging.warning(package)
    d,__,__,__,__,__ = np.shape(image)
    logging.warning("d data: \n")
    logging.warning(d)
    small_array = np.asarray(image[0:10])
    np.save("shrink_2.npy", small_array)


def read_input(path=None):
    """Reads input from dataset and labels."""
    # load training data
    image = np.load("./X_light.npy", mmap_mode="r")
    # shape of training data: shape = [number of package, batch_size, timesteps, height, width, channels]
    # channels -> channel[0]= east-ward wind; channels[1]=north-ward wind; channel[2]=precipitation
    # extract channels? YES
    channels = image[:,:,:,:,:,0:3]

    # load labels
    label=np.load("./Y_light.npy")

    # TODO: is d the number of packages? YES, helps with preprocessing
    d,__,__,__,__,__ = np.shape(image)
    print "d:", d

    # Splitting data

    # Test set ~20% of data; about 28
    te_image=np.asarray(channels[0:int(d*0.2)])
    te_label=np.asarray(label[0:int(d*0.2)])

    # Training set ~80% of data; controls the number of training steps in one epoch
    tr_image=np.asarray(channels[int(d*0.80):d-10])
    tr_label=np.asarray(label[int(d*0.80):d-10])

    # Validation set 10 objects
    va_image=np.asarray(channels[d-10:d])
    va_label=np.asarray(label[d-10:d])

    # Save test set to numpy array
    np.save("X_test.npy",te_image)
    np.save("Y_test.npy",te_label)

    # print the shape of the data
    print(np.shape(tr_image),np.shape(tr_label),np.shape(te_image),np.shape(te_label),np.shape(va_image),np.shape(va_label))
    return tr_image,tr_label,te_image,te_label,va_image,va_label

def main():
    # shrink_data()
    read_input()

if __name__ == '__main__':
    main()
