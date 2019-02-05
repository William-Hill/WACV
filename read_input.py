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


def read_input(path):
    """Reads input from dataset and labels."""
    # load training data
    image = np.load("./X_light.npy", mmap_mode="r")
    # shape of training data: shape = [number of package, batch_size, timesteps, height, width, channels]
    # channels -> channel[0]= east-ward wind; channels[1]=north-ward wind; channel[2]=precipitation
    # extract channels?
    channels = image[:,:,:,:,:,0:3]

    # load labels
    label=np.load("./Y_light.npy")
    d,__,__,__,__,__ = np.shape(image)
    te_image=np.asarray(image[0:int(d*0.2)])
    te_label=np.asarray(label[0:int(d*0.2)])
    tr_image=np.asarray(image[int(d*0.80):d-10])
    tr_label=np.asarray(label[int(d*0.80):d-10])
    va_image=np.asarray(image[d-10:d])
    va_label=np.asarray(label[d-10:d])
    np.save("X_test.npy",te_image)
    np.save("Y_test.npy",te_label)
    print(np.shape(tr_image),np.shape(tr_label),np.shape(te_image),np.shape(te_label),np.shape(va_image),np.shape(va_label))
    return tr_image,tr_label,te_image,te_label,va_image,va_label

def main():
    shrink_data()

if __name__ == '__main__':
    main()
