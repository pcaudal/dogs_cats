#!/usr/bin/python3
# -*- coding: utf8 -*-

"""Convolutional Neural Networks for Dogs and Cats pics."""

# from prj_function import create_dictionary, model_accuracy, model_loss
from prj_function import model_acc_loss
from keras.models import model_from_json
import json
# from data_generator import DataGeneratorPNG
import data_generator as dg

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout 
# from keras.layers import Flatten
# from keras.layers.convolutional import Conv2D 
# from keras.layers.convolutional import MaxPooling2D
# from keras.callbacks import TensorBoard
# from pandas.tests.indexes.datetimes.test_tools import epochs

path_project = "/home/tostakit/Téléchargements/dogsvscats/"
path_data = "/home/tostakit/Téléchargements/dogsvscats/sample/"
path_tb_log = "/media/tostakit/Partage/dev/pyprojects/1.0-data_scientist/dogs_cats/tb_logs/"

# Input Image Dimensions
img_rows, img_cols = 150, 150
# Number of Convolutional Filters to use
batch_size = 2
# Number of classes (for mnist, 10 (0 to 9)
n_classes = 2
# Number of channels is 3 for RGB png pics
n_channels = 3
# Number of Convolutional Filters to use
nb_filters = 32
# Number of epochs
n_epochs = 10
# Thread used
multi_thread = False
# Number of thread (1 if multi_thread = False)
if multi_thread:
    n_thread = 4
else:
    n_thread = 1    

# Parameters for the DataGenerator method
params = {'dim': (img_rows, img_cols),
          'batch_size': batch_size,
          'n_classes': n_classes,
          'n_channels': n_channels,
          'shuffle': False}

with open(path_project+'partition.json', 'r') as json_file:  
    partition = json.load(json_file)
with open(path_project+'labels.json', 'r') as json_file:  
    labels = json.load(json_file)
with open(path_project+'model.json', 'r') as json_file:  
    loaded_model_json = json.load(json_file)
loaded_model = model_from_json(loaded_model_json)
# # load json and create model
# json_file = open(path_project+'model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(path_project+"model.h5")
print("Loaded model from disk")

validation_generator = dg.DataGeneratorJPG(partition['validation'], labels, path_data, **params)

print("evaluation.py : History Fit")
print(loaded_model.history.keys())

print("evaluation.py : Model accuracy and loss")

model_acc_loss(loaded_model.history['acc'], loaded_model.history['val_acc'],
               loaded_model.history['loss'], loaded_model.history['val_loss'],
               n_epochs)

# Charger partition.csv et labels.csv
#validation_generator = dg.DataGeneratorJPG(partition['validation'], labels, path_data, **params)
 
print("evaluation.py : Predict Model")
predict = loaded_model.predict_generator(generator=validation_generator,
                                 use_multiprocessing=multi_thread,
                                 workers=n_thread,
                                 verbose=1)
 
print("evaluation.py : Evaluate Model")
scores = loaded_model.evaluate_generator(generator=validation_generator,
                         use_multiprocessing=multi_thread,
                         workers=n_thread)
#                        workers = n_thread,
#                        verbose = 0)
print("Perte: %.2f Erreur: %.2f%%" % (scores[0], 100 - scores[1] * 100))
