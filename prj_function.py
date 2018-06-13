#!/usr/bin/python3
# -*- coding: utf8 -*-

"""Convolutional Neural Networks for Dogs and Cats."""

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
import json
import itertools

from keras.models import Sequential, model_from_json
from keras.layers import Dense
#from keras.layers import Dropout 
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D

def design_model(img_rows, img_cols, n_channels, n_filters, n_classes):

    print("    fct design_model : design model creation")
    # Design model
    # CNN Initialisation 
    model = Sequential()
    
    # First Convolution layer (for dogs ands cats : input_shape = (50, 50, 3)
    model.add(Conv2D(n_filters, (3, 3), input_shape = (img_rows, img_cols, n_channels), activation = 'relu'))
    
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Second convolutional layer
    model.add(Conv2D(n_filters, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Flatten())
    
    # Full connection
    model.add(Dense(units = 4 * n_filters, activation = 'relu'))
    #model.add(Dropout (0.50))
    model.add(Dense(units = n_classes, activation = 'sigmoid'))
    
    # Compiling the CNN
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model


def save_model_history(model, history, path_json, json_model, h5_weigth, history_file):

    print("    fct save_model_history : Saved model to disk")
    model_json = model.to_json()
    with open(path_json+json_model, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    print("    fct save_model_history : Saved weights to disk")
    model.save_weights(path_json+h5_weigth)
    
    # print("dl_dogs_cats.py : History Fit keys")
    # print(history.history.keys())
    
    print("    fct save_model_history : Saved hystory.history to disk history_model_complete.json")
    with open(path_json+history_file, "w") as outfile:
        json.dump(history, outfile)

def load_model_history(path_json, json_model, h5_weigth, history_file):

    print("    fct load_model_history : Load model from disk")
    # load json and create model
    with open(path_json+json_model, 'r') as json_file:  
        # loaded_model_json = json.load(json_file)
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    print("    fct load_model_history : Load weights from disk")
    loaded_model.load_weights(path_json+h5_weigth)

    # print("    fct load_model_history : loaded model compilation")
    # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    print("    fct load_model_history : Load history from disk")
    with open(path_json+history_file, "r") as json_file:
        loaded_history = json.load(json_file)
    
    return loaded_model, loaded_history

def load_model(path_json, json_model, h5_weigth):

    print("    fct load_model : Load model with weights from disk and compile")
    # load json and create model
    with open(path_json+json_model, 'r') as json_file:  
        # loaded_model_json = json.load(json_file)
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    print("    fct load_model_history : Load weights from disk")
    loaded_model.load_weights(path_json+h5_weigth)
    
    print("    fct load_model_history : Compile model")
    # model.add(Dense(10, kernel_initializer="uniform", activation="linear"))
    loaded_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    return loaded_model


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def pred_eval_model(model, validation_generator, b_cluster, n_processor):
 
    print("    fct pred_eval_model : Predict Model")
    predict = model.predict_generator(generator=validation_generator,
                                     use_multiprocessing=b_cluster,
                                     workers=n_processor,
                                     verbose=1)
     
    print("    fct pred_eval_model : Evaluate Model")
    scores = model.evaluate_generator(generator=validation_generator,
                             use_multiprocessing=b_cluster,
                             workers=n_processor)

    print("    fct pred_eval_model : Evaluation result :")
    print("        ----> Perte: %.2f Erreur: %.2f%%" % (scores[0], 100 - scores[1] * 100))

    return predict, scores


def create_dictionary(labels_csv_file, path='data/', ratio=0.2):

    print("    fct create_dictionary : dict_labels creation")
    # labels
    # {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}
    list_values = list(map(int, genfromtxt(path + labels_csv_file, delimiter=',')))
    nb_img = len(list_values)
    list_keys = list(map(str, range(nb_img)))
    dict_labels = dict(zip(list_keys, list_values))
      
    print("    fct create_dictionary : dict_partition creation")
    # partition
    # {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
    nb_img_test = round(ratio * nb_img)
    nb_img_train = nb_img - nb_img_test
    
    dict_partition = dict({'train':list(map(str, range(nb_img_train))), 'validation':list(map(str, range(nb_img_train, nb_img_train + nb_img_test)))})
    
    return dict_partition, dict_labels

def create_sub_dictionary(df_labels, ratio=0.2):

    print("    fct create_sub_dictionary : dict_labels creation")
    # labels
    # {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}
    list_values = df_labels[0].tolist()
    list_keys = list(map(str, df_labels['img_id'].tolist()))
    dict_labels = dict(zip(list_keys, list_values))
    nb_img = len(list_values)
      
    print("    fct create_sub_dictionary : dict_partition creation")
    # partition
    # {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
    nb_img_test = round(ratio * nb_img)
    nb_img_train = nb_img - nb_img_test
    
    dict_partition = dict({'train':list_keys[0:nb_img_train], 'validation':list_keys[nb_img_train:nb_img_train + nb_img_test]})
    
    return dict_partition, dict_labels
