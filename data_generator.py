# -*- coding: utf-8 -*-
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

import numpy as np
import keras
import matplotlib.image as mpimg
from keras.callbacks import TensorBoard
import tensorflow as tf
import os
import cv2


class DataGeneratorPNG(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, path_data,
                 batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        # self.path = "/media/tostakit/Partage/google-drive/data_scientist/mnist_data/data/"
        self.path = path_data
        self.img_ext = '.png'
        # self.compteur = 0
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.n_channels != 0:
            X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        else:
            X = np.zeros((self.batch_size, *self.dim, 1))
#        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size), dtype=int)

        if self.n_channels != 0:
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                X[i, ] = mpimg.imread(self.path + ID + self.img_ext)
                # Store class
                y[i] = self.labels[ID]
        else:
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                ar_img = mpimg.imread(self.path + ID + self.img_ext)
                # ar_img = (ar_img * 255 ).astype(np.uint8)
                temp = np.zeros((self.dim[0], self.dim[1], 1))
                temp[:, :, 0] = ar_img
                X[i, ] = temp
                # Store class
                y[i] = self.labels[ID]
                
                # self.compteur += 1
                # print("class_data_generator : compteur de data_generation = ", self.compteur)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
class DataGeneratorJPG(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, path_data,
                 batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        # self.path = "/media/tostakit/Partage/google-drive/data_scientist/mnist_data/data/"
        self.path = path_data
        self.img_ext = '.jpg'
        # self.compteur = 0
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.n_channels != 0:
            X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        else:
            X = np.zeros((self.batch_size, *self.dim, 1))
#        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size), dtype=int)

        if self.n_channels != 0:
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                X[i, ] = cv2.imread(self.path + ID + self.img_ext)
                # Store class
                y[i] = self.labels[ID]
        else:
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                ar_img = cv2.imread(self.path + ID + self.img_ext)
                # ar_img = (ar_img * 255 ).astype(np.uint8)
                temp = np.zeros((self.dim[0], self.dim[1], 1))
                temp[:, :, 0] = ar_img
                X[i, ] = temp
                # Store class
                y[i] = self.labels[ID]
                
                # self.compteur += 1
                # print("class_data_generator : compteur de data_generation = ", self.compteur)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    

# Module from :
# https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure/48393723?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa 
class TrainValTensorBoard(TensorBoard):
    
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
