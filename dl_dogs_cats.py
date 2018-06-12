#!/usr/bin/python3
# -*- coding: utf8 -*-


"""Voir inférence bayesienne pour des dataset très petit : b-testing, pymc3"""
"""Convolutional Neural Networks for Dogs and Cats pics."""

import data_generator as dg
#from prj_function import create_dictionary, create_sub_dictionary, model_acc_loss
#from prj_function import design_model, save_model_history, load_model, pred_eval_model
from prj_function import design_model, save_model_history
import json
import os
import shutil as sh

import numpy as np
#import pandas as pd
#import time
import matplotlib.pyplot as plt

#from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# fix random seed for reproducibility
# np.random.seed(7)

print("dl_dogs_cats.py : Download parameters json file")

path_project = os.path.dirname(os.path.abspath(__file__))
path_json = path_project+"/json/"

with open(path_json+'config_project.json', 'r') as json_file:  
    dict_params = json.load(json_file)

##############################################################################
print("dl_dogs_cats.py : Settings Inputs")

# Model json file name to save
file_model_json = dict_params['file_name']['model_json']
# Weightd of the nerons h5 file name to save
file_weigths_h5 = dict_params['file_name']['weigths_h5']
# Model history json file name to save
file_history_json = dict_params['file_name']['history_json']
# Image status json file name to save
file_img_status_json = dict_params['file_name']['img_status']
# Name of the labels csv file
labels_file = dict_params['file_name']['labels_file']
# Config project file name to save
config_project = dict_params['file_name']['config_project']
# Choose True if the model is trained for the first time train or retrained with the complete dataset
b_complete_train = dict_params['b_complete_train']
# Directory where the trained images are saved in folder cats and dogs
path_data_train = dict_params['path']['data_train']
# Directory where the test images are saved in folder cats and dogs
path_data_test = dict_params['path']['data_test']

path_data = dict_params['path']['data']
# Directory where the tensorboard logs are saved
path_tb_log = dict_params['path']['tb_log']
# Path where json files are saved in retrain mode
path_json_retrain = dict_params['path']['json_retrain']
# Input Image Dimensions
img_rows, img_cols = dict_params['img']['rows'], dict_params['img']['cols']
# Number of channels is 3 for RGB png pics
n_channels = dict_params['img']['n_channels']
# Number of image in the batch
batch_size = dict_params['batch_size']
# Number of classes (for mnist, 10 (0 to 9), for dogs and cats : 2 (0 or 1))
n_classes = dict_params['n_classes']
# Number of Convolutional Filters to use
n_filters = dict_params['n_filters']
# Number of epochs
n_epochs = dict_params['n_epochs']
# shuffle
b_shuffle = dict_params['b_shuffle']
# Cluster of processor used or not
b_cluster = dict_params['b_cluster']
# Number of processor used (1 if cluster_used = False)
n_processor = dict_params['n_processor']
# Ratio du nombre d'images de validation
ratio_validation_img = dict_params['ratio_validation_img']
n_data_test = dict_params['data']['test']
n_data_train = dict_params['data']['train']

##############################################################################

sh.rmtree(path_tb_log)
os.makedirs(path_tb_log+'training/', mode=0o777)
os.makedirs(path_tb_log+'validation/', mode=0o777)

# Parameters for the DataGenerator method
params = {'dim': (img_rows, img_cols),
          'batch_size': batch_size,
          'n_classes': n_classes,
          'n_channels': n_channels,
          'shuffle': b_shuffle}

# If b_first_train, it means that the train model is trained for the first time
if b_complete_train:
    
    print("dl_dogs_cats.py : ******* Process a train model for the first time")
    print("dl_dogs_cats.py : Design the complete model")
    model = design_model(img_rows, img_cols, n_channels, n_filters, n_classes)
    
    print("dl_dogs_cats.py : Batch creation")
    # https://keras.io/preprocessing/image/
    # https://gist.github.com/RyanAkilos/3808c17f79e77c4117de35aa68447045
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_generator = train_datagen.flow_from_directory(path_data_train,
                                                           classes=['Dogs', 'Cats'],
                                                           target_size = (img_rows, img_cols),
                                                           batch_size = batch_size,
                                                           class_mode = 'categorical')
#                                                         class_mode = 'binary')

    validation_generator = train_datagen.flow_from_directory(path_data_test,
                                                             classes=['Dogs', 'Cats'],
                                                             target_size = (img_rows, img_cols),
                                                             batch_size = batch_size,
                                                             class_mode = 'categorical')
#                                                            class_mode = 'binary')
    
    print("dl_dogs_cats.py : Fit Complete Model with tensorboard callbacks")
    print("  - open a terminal and launch the following commands :")
    print("      $ cd ",path_project)
    print("      $ tensorboard --logdir ./tb_logs")
    print("  - open your web browser and enter the address below")
    print("      htttp://localhost:6006")
    # Train model on dataset
    
    history = model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        steps_per_epoch = n_data_train // batch_size,
                        epochs = n_epochs,
                        validation_steps = n_data_test // batch_size,
                        use_multiprocessing=b_cluster,
                        workers=n_processor,
#                         verbose=2)
                        verbose=2,
                        callbacks=[dg.TrainValTensorBoard(log_dir=path_tb_log, write_graph=False)])

    # serialize model to JSON
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    print("dl_dogs_cats.py : Saved complete model and history to disk")
    save_model_history(model, history, path_json,
                       file_model_json,
                       file_weigths_h5,
                       file_history_json)
 
    print("dl_dogs_cats.py : Predict Model")
    Y_pred = model.predict_generator(generator=validation_generator,
                                     steps = n_data_test // batch_size+1,
                                     use_multiprocessing=b_cluster,
                                     workers=n_processor,
                                     verbose=1)

    y_pred = np.argmax(Y_pred, axis=1)
    print('dl_dogs_cats.py : Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))
    print('dl_dogs_cats.py : Classification Report')
    target_names = ['Cats', 'Dogs']
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
    
    print("dl_dogs_cats.py : Evaluate Model")
    scores = model.evaluate_generator(generator=validation_generator,
                                     steps = n_data_test // batch_size+1,
                                     use_multiprocessing=b_cluster,
                                     workers=n_processor)

    print("dl_dogs_cats.py : Evaluation result :")
    print("        ----> Perte: %.2f Erreur: %.2f%%" % (scores[0], 100 - scores[1] * 100))
    

    print("dl_dogs_cats.py : Complete train model accuracy and loss figure with matplotlib")
#     model_acc_loss(history.history['acc'], history.history['val_acc'],
#                     history.history['loss'], history.history['val_loss'],
#                     n_epochs)
    with open(path_json+file_history_json, 'r') as json_file:  
        history = json.load(json_file)

    x_epochs = range(n_epochs)
    fig = plt.figure()
    plt.gcf().subplots_adjust(hspace=0.5)
    ax1 = fig.add_subplot(211)
    ax1.plot(x_epochs, history['acc'], label='Train', color='blue')
    ax1.plot(x_epochs, history['val_acc'], label='Test', color='red')
    ax1.set_xlim([0, n_epochs])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.legend(loc=4)
    ax2 = fig.add_subplot(212)#, sharex=ax1)
    ax2.plot(x_epochs, history['loss'], label='Train', color='blue')
    ax2.plot(x_epochs, history['val_loss'], label='Test', color='red')
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend (loc='best')
    fig.show()


# If b_first_train is false, it means that the train model is loaded and retrained with new images
else:
    pass
    
#     print("dl_dogs_cats.py : ****** Start of the process to retain a model")
#     print("dl_dogs_cats.py : Load the model without the weights from disk")
#     print("dl_dogs_cats.py : the origin of the loaded_model comes from the first part of this script,")
#     print("dl_dogs_cats.py : when b_complete_train was set at True")
#     # load json and create model
#     # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#     with open(path_json+file_model_json, 'r') as json_file:  
#         # loaded_model_json = json.load(json_file)
#         loaded_model_json = json_file.read()
#     loaded_model = model_from_json(loaded_model_json)
#      
#     print("dl_dogs_cats.py : Compile loaded model")
#     # model.add(Dense(10, kernel_initializer="uniform", activation="linear"))
#     loaded_model.compile(optimizer='adam',
#                    loss='categorical_crossentropy',
#                    metrics=['accuracy'])
# 
# #     print("dl_dogs_cats.py : Design the complete model")
# #     loaded_model = design_model(img_rows, img_cols, n_channels, n_filters, n_classes)
#     
#     # Load partition and labels dict from json files on the disk
#     # http://stackabuse.com/reading-and-writing-json-to-a-file-in-python/
#     print("dl_dogs_cats.py : Load partition and labels dictionaries from json files")
#     print("dl_dogs_cats.py : As loaded model, these json files were saved at the of of the ")
#     print("dl_dogs_cats.py : first part of this script, when b_complete_train was set at True")
#     with open(path_json+'partition.json', 'r') as json_file:  
#         partition = json.load(json_file)
#     with open(path_json+'labels.json', 'r') as json_file:  
#         labels = json.load(json_file)
#     with open(path_json+file_img_status_json, 'r') as json_file:  
#         img_status = json.load(json_file)
#     
#     #n_img_labels = len(labels)
#     n_img = img_status['n_img']
#     print("dl_dogs_cats.py : Total number of cats and dogs images (n_img) : ", n_img)
# 
#     # Number of images treated by iteration to retrain the model
#     n_img_package = dict_params['n_img_package']
#     print("dl_dogs_cats.py : Number of images treated by iteration to retrain the model (n_img_package) : ", n_img_package)
#     
#     n_retrain = n_img//n_img_package
#     print("dl_dogs_cats.py : The model have to be retrain  (n_img/n_img_package) : ", n_retrain)
#     
#     df_labels = pd.read_csv(path_data + labels_file, sep=',', header=None)
#     df_labels['img_id']=df_labels.index
#     total_img = len(df_labels)
#     index_start = 0
#     index_stop = n_img_package
#     
#     t_start = time.time()
#     t_total = 0
#     for i in range(n_retrain):
#         print("dl_dogs_cats.py : ========> a stack of ", n_img_package, " images are going to be processed...")
#         # df_labels cut (columns : 0 for labels and 'img_id' for the name pf the img
#         df = df_labels[:][index_start:index_stop]
#         
#         print("dl_dogs_cats.py : Creation of the dict partition and labels")
#         # Datasets
#         partition, labels = create_sub_dictionary(df, ratio=ratio_validation_img)
#         
#         
#         print("dl_dogs_cats.py : Object creation -> training_generator and validation_generator with class_data_generator")
#         # Generators
#         training_generator = dg.DataGeneratorJPG(partition['train'], labels, path_data, **params)
#         validation_generator = dg.DataGeneratorJPG(partition['validation'], labels, path_data, **params)
# 
#         print("dl_dogs_cats.py : Fit Partial Model with tensorboard callbacks")
#         print("  - open a terminal and launch the following commands :")
#         print("      $ cd ",path_project)
#         print("      $ tensorboard --logdir ./tb_logs")
#         print("  - open your web browser and enter the address below")
#         print("      htttp://localhost:6006")
#         # Train model on dataset
#         history = loaded_model.fit_generator(generator=training_generator,
#                             validation_data=validation_generator,
#                             use_multiprocessing=b_cluster,
#                             workers=n_processor,
#                             epochs=n_epochs,
#                             verbose=2,
#                             callbacks=[dg.TrainValTensorBoard(log_dir=path_tb_log, write_graph=False)])
# 
#         print("dl_dogs_cats.py : Partial model accuracy and loss figure with matplotlib")
#         model_acc_loss(history.history['acc'], history.history['val_acc'],
#                         history.history['loss'], history.history['val_loss'],
#                         n_epochs)
#     
#         print("dl_dogs_cats.py : Predict and evaluation the partial model")
#         predict, scores = pred_eval_model(loaded_model, validation_generator, b_cluster, n_processor)
#     
#         print("dl_dogs_cats.py : Saved complete model and history to disk")
#         save_model_history(loaded_model, history, path_json_retrain,
#                            str(i)+'_loaded_'+file_model_json,
#                            str(i)+'_loaded_'+file_weigths_h5,
#                            str(i)+'_loaded_'+file_history_json)
#     
#         print("dl_dogs_cats.py : Load model from disk for a new retrain")
#         loaded_model = load_model(path_json_retrain,
#                                    str(i)+'_loaded_'+file_model_json,
#                                    str(i)+'_loaded_'+file_weigths_h5)
#         
#         ellapsed = time.time() - t_start
#         t_total += ellapsed
#         print('ddl_dogs_cats.py : Retrain (done/finale) ', i+1,
#               '/', n_retrain,
#               ', takes %.2fs ' %ellapsed,
#               'for a total time of %.2fs.' %t_total)
#         print("dl_dogs_cats.py : status (process image / total of image) : ", index_stop, '/', total_img)
#         
#         index_start += n_img_package
#         index_stop += n_img_package


    print("dl_dogs_cats.py : ****** End of the process to retain a model")

