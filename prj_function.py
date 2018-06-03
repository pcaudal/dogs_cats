#!/usr/bin/python3
# -*- coding: utf8 -*-

"""Convolutional Neural Networks for Dogs and Cats."""

from numpy import genfromtxt
import matplotlib.pyplot as plt


def create_dictionary(path='data/', ratio=0.2):

    print("fct create_dictionary : Début de la création de dict_labels")
    # labels
    # {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}
    list_values = list(map(int, genfromtxt(path + 'png_labels.csv', delimiter=',')))
    nb_img = len(list_values)
    list_keys = list(map(str, range(nb_img)))
    dict_labels = dict(zip(list_keys, list_values))
    print("fct create_dictionary : dict_labels est créé")
      
    print("fct create_dictionary : Début de la création de dict_partition")
    # partition
    # {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
    nb_img_test = round(ratio * nb_img)
    nb_img_train = nb_img - nb_img_test
    
    dict_partition = dict({'train':list(map(str, range(nb_img_train))), 'validation':list(map(str, range(nb_img_train, nb_img_train + nb_img_test)))})
    print("fct create_dictionary : dict_partition est créé")
    
    return dict_partition, dict_labels


def model_acc_loss(acc, val_acc, loss, val_loss, n_epochs):
    x_epochs = range(n_epochs)
    fig = plt.figure()
    plt.gcf().subplots_adjust(hspace=0.5)
    ax1 = fig.add_subplot(211)
    ax1.plot(x_epochs, acc, label='Train', color='blue')
    ax1.plot(x_epochs, val_acc, label='Test', color='red')
    ax1.set_xlim([0, 9])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.legend(loc=4)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(x_epochs, loss, label='Train', color='blue')
    ax2.plot(x_epochs, val_loss, label='Test', color='red')
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend (loc='best')
    fig.show()


def model_accuracy(acc, val_acc, n_epochs):
    x_epochs = range(n_epochs)
    plt.plot(x_epochs, acc, label='Train', color='blue')
    plt.plot(x_epochs, val_acc, label='Test', color='red')
    plt.axis([0, 9, 0, 1.0])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc=4)
    plt.show()


def model_loss(loss, val_loss, n_epochs):
    x_epochs = range(n_epochs)
    plt.plot(x_epochs, loss, label='Train', color='blue')
    plt.plot(x_epochs, val_loss, label='Test', color='red')
    plt.axis([0, 9, 0, 1.0])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()

