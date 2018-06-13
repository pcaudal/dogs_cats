# -*- coding: utf-8 -*-
"""
Paramters for Images recongnition

@author: Philippe Caudal
"""

import json
import os

#path_project = "/home/tostakit/Téléchargements/dogsvscats/"
path_project = os.path.dirname(os.path.abspath(__file__))
path_orig_img = "/media/tostakit/Partage/dev/datasets/dogsvscats/"
path_data = path_orig_img+"data/"
path_data_train = path_data+"train/"
path_data_test = path_data+"test/"
path_data_train_dogs = path_data_train+"Dogs/"
path_data_test_dogs = path_data_test+"Dogs/"
path_data_train_cats = path_data_train+"Cats/"
path_data_test_cats = path_data_test+"Cats/"
path_tb_log = path_project+"/tb_logs/"
path_json = path_project+"/json/"
# Path where json files are saved in retrain mode
path_json_retrain = path_json+'retrain/'

# path_project = "D:\\philippe\\France\\travail\\data science\\ORT France\\langage python\\projet\\dogs_cats\\"
# path_data = "D:\\philippe\\France\\travail\\data science\\ORT France\\langage python\\projet\\dogs_cats\\sample\\"
# path_tb_log = "D:\\philippe\\France\\travail\\data science\\ORT France\\langage python\\projet\\dogs_cats\\tb_logs\\"

print("param_json.py : Chemin du projet : ",os.path.dirname(os.path.abspath(__file__)))

# Choose True if the model has to train the complete dataset
b_complete_train = False
# Number of dog and cat images trained at the first launch
# """ Makes sure that the total number images modulo n_img_package = 0"""
# n_img_package = 10
# Number of image in the batch
""" Makes sure that the total number images modulo batch_size = 0"""
batch_size = 32
# Name of the labels csv file
labels_file = 'img_labels.csv'
# Name of the json file which save the status of the images files used to train
img_status = 'img_status.json'
# Model json file name to save
model_json = "model.json"
# Weightd of the nerons h5 file name to save
weigths_h5 = "weights.h5"
# Model history json file name to save
history_json = "history.json"
# Config project file name to save
config_project = "config_project.json"
# Input Image Dimensions : width, length and number of channels is 3 for RGB png pics
img_rows, img_cols, n_channels = 128, 128, 3
# Number of classes (for mnist, 10 (0 to 9)
n_classes = 2
# Number of Convolutional Filters to use
n_filters = 64
# Number of epochs
n_epochs = 1
# Cluster of processor used or not
b_cluster = False
# Number of processor used (1 if cluster_used = False)
if b_cluster:
    n_processor = 4
else:
    n_processor = 1    
# Shuffle
b_shuffle = True
# Images type
img_ext = "jpg"
# Ratio du nombre d'images de validation (soit 250 images de chiens et 250 images de chats)
#ratio_validation_img = 0.01
ratio_validation_img = 0.1
#total sample
data_total = 4000
#Number sample test
data_test = round(data_total*ratio_validation_img)
#Number sample train
data_train = data_total - data_test


print("param_json.py : creation of the parameters dictionary")

path = {"project":path_project,
        'orig_img':path_orig_img,
        'data':path_data,
        "data_train":path_data_train,
        "data_test":path_data_test,
        "data_train_dogs":path_data_train_dogs,
        "data_train_cats":path_data_train_cats,
        "data_test_dogs":path_data_test_dogs,
        "data_test_cats":path_data_test_cats,
        "tb_log":path_tb_log,
        "json":path_json,
        'json_retrain':path_json_retrain}

data = {'total':data_total,
        'test':data_test,
        'train':data_train}

file_name = {'labels_file':labels_file,
             'config_project': config_project,
             'model_json': model_json,
             'weigths_h5': weigths_h5,
             'history_json': history_json,
             'img_status': img_status}


img = {"rows":img_rows, "cols":img_cols, "n_channels":n_channels, "extension":img_ext}

parameters = {"path":path,
              'file_name':file_name,              
              'data':data,
              "img":img,
              'ratio_validation_img':ratio_validation_img,
#              'n_img_package':n_img_package,
              "b_complete_train":b_complete_train,
              "batch_size":batch_size,
              'n_classes': n_classes,
              "n_filters":n_filters,
              "n_epochs":n_epochs,
              'b_cluster':b_cluster,
              "n_processor":n_processor,
              'b_shuffle': b_shuffle}

# http://stackabuse.com/reading-and-writing-json-to-a-file-in-python/
print("config_project.py : save the parameters dictionary as config_project.json, in the folder :")
print("        ", path_json)

with open(path_json+'config_project.json', 'w') as outfile:  
    json.dump(parameters, outfile)
    
