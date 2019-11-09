import pandas as pd
import numpy as np
import h5py
from skimage import io, transform
import matplotlib.pyplot as plt

def load_data(fn, path='./datasets/', sep=','):
    data = pd.read_csv(path + fn, sep=sep)
    return data


    
    
def load_cat_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def predict_image(clf, fname, num_px, classes):
    image_file = "images/" + fname
    image = io.imread(image_file) / 255.
    resized_image = transform.resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3))
    my_predicted_image = clf.predict(resized_image)

    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + 
          classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")