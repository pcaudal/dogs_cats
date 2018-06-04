#!/usr/bin/python3
# -*- coding: utf8 -*-

"""Convolutional Neural Networks for Dogs and Cats pics."""

# from data_generator import DataGeneratorPNG
import data_generator as dg
# from prj_function import create_dictionary, model_accuracy, model_loss
from prj_function import create_dictionary, model_acc_loss
###### import pour affichage des résultats
# from report_result import visu_img_non_predict, report_conf_mat
###################################################################
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout 
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D
# from keras.callbacks import TensorBoard
# from pandas.tests.indexes.datetimes.test_tools import epochs

# from sklearn.metrics import classification_report, accuracy_score

print("dl_dogs_cats.py : Chargement des variables d'entrée")
# link to to images et labels repository (googledrive)
# path_data = "/media/tostakit/Partage/google-drive/data_scientist/dogs_cats/data/"
# path_data = "/home/tostakit/Téléchargements/dogsvscats/"

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

print("dl_dogs_cats.py : Création des dictionnaires partition et labels")
# Datasets
partition, labels = create_dictionary(path=path_data)

# Save partition and labels dict in json files
# http://stackabuse.com/reading-and-writing-json-to-a-file-in-python/
with open(path_project+'partition.json', 'w') as outfile:  
    json.dump(partition, outfile)
with open(path_project+'labels.json', 'w') as outfile:  
    json.dump(labels, outfile)


print("dl_dogs_cats.py : Création des objets training_generator et validation_generator avec la class_data_generator")
# Generators
training_generator = dg.DataGeneratorJPG(partition['train'], labels, path_data, **params)
validation_generator = dg.DataGeneratorJPG(partition['validation'], labels, path_data, **params)

print("dl_dogs_cats.py : Design model")
# Design model
model = Sequential()
model.add(Conv2D(nb_filters,
                 kernel_size=(5, 5),
                 strides=1,
                 activation="relu",
                 input_shape=(img_rows, img_cols, n_channels),
#                 data_format='channels_first'))
                 data_format='channels_last'))

model.add(Conv2D(2 * nb_filters,
                 kernel_size=(5, 5),
                 strides=1,
                 activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(4 * nb_filters, activation="relu"))
model.add(Dropout (0.50))

model.add(Dense(n_classes,
                kernel_initializer="uniform",
                activation="softmax"))

# model.add(Dense(10, kernel_initializer="uniform", activation="linear"))
model.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# print("dl_dogs_cats.py : Tensorboard init")
# tensor_board = TensorBoard(log_dir=path_tb_log,
#                            histogram_freq=0,
#                            write_graph=True,
#                            write_images=True)

print("dl_dogs_cats.py : Fit Model with tensorboard callbacks")
print("  - open a terminal, cd in the work space and launch the following command :")
print("      tensorboard --logdir ./tb_logs")
print("  - open your web browser and enter the address below")
print("      htttp://localhost:6006")
# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=multi_thread,
                    workers=n_thread,
                    epochs=n_epochs,
                    verbose=2,
                    callbacks=[dg.TrainValTensorBoard(log_dir=path_tb_log, write_graph=False)])

# serialize model to JSON
model_json = model.to_json()
with open(path_project+"model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(path_project+"model.h5")
print("dl_dogs_cats.py : Saved model to disk")


print("dl_dogs_cats.py : History Fit")
print(history.history.keys())

print("dl_dogs_cats.py : Model accuracy and loss")
# model_accuracy(history.history['acc'], history.history['val_acc'], n_epochs)
# model_loss(history.history['loss'], history.history['val_loss'], n_epochs)
model_acc_loss(history.history['acc'], history.history['val_acc'],
               history.history['loss'], history.history['val_loss'],
               n_epochs)

 
print("dl_dogs_cats.py : Predict Model")
predict = model.predict_generator(generator=validation_generator,
                                 use_multiprocessing=multi_thread,
                                 workers=n_thread,
                                 verbose=1)
 
print("dl_dogs_cats.py : Evaluate Model")
scores = model.evaluate_generator(generator=validation_generator,
                         use_multiprocessing=multi_thread,
                         workers=n_thread)
#                        workers = n_thread,
#                        verbose = 0)
print("Perte: %.2f Erreur: %.2f%%" % (scores[0], 100 - scores[1] * 100))

