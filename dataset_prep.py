#!/usr/bin/python3
# -*- coding: utf8 -*-

# dataset_prep.py : launch once to format the images and create the
# img_labels.json which gives the labels (0 for a dog, 1 for a cat)
# and the id of the line in the label file, the id of the image.
# labels : 0 for a cat, and 1 for a dog

import shutil as sh
import os
# import random
# import csv
# import numpy as np
# import pandas as pd
import cv2  # installation : opencv
import json
import time
# import copy

from glob import glob
from builtins import enumerate

print("dataset_prep.py : Download init parameters from json file")
path_project = os.path.dirname(os.path.abspath(__file__))
path_json = path_project+"/json/"

with open(path_json+'config_project.json', 'r') as json_file:  
    dict_params = json.load(json_file)

print("dataset_prep.py : set variables")
# Directory where the images are saved
path_orig_img = dict_params['path']['orig_img']
path_data = dict_params['path']['data']
path_data_train_dogs = dict_params['path']['data_train_dogs']
path_data_train_cats = dict_params['path']['data_train_cats']
path_data_test_dogs = dict_params['path']['data_test_dogs']
path_data_test_cats = dict_params['path']['data_test_cats']

height = dict_params['img']['rows']
width = dict_params['img']['cols']
n_channels = dict_params['img']['n_channels']
img_ext = "."+dict_params['img']['extension']
# Name of the labels csv file
labels_file = dict_params['file_name']['labels_file']

ratio_validation_img = dict_params['ratio_validation_img']

files_dog_path = os.path.join(path_orig_img, 'dog.*'+img_ext)
files_cat_path = os.path.join(path_orig_img, 'cat.*'+img_ext)
files_dog = sorted(glob(files_dog_path))
files_cat = sorted(glob(files_cat_path))

n_dog_img = len(files_dog)
n_dog_img_test = round(ratio_validation_img*n_dog_img)
n_dog_img_train = n_dog_img - n_dog_img_test

n_cat_img = len(files_cat)
n_cat_img_test = round(ratio_validation_img*n_cat_img)
n_cat_img_train = n_cat_img - n_cat_img_test

files = []
files.extend(files_dog)
files.extend(files_cat)
n_img = len(files)


# dict_params['data']['total'] = n_img
# dict_params['data']['test'] = n_dog_img_test + n_cat_img_test
# dict_params['data']['train'] = n_dog_img_train + n_cat_img_train
# print("dataset_prep.py : save the parameters dictionary as config_project.json, in the folder :")
# print("        ", path_json)
# with open(path_json+'config_project.json', 'w') as outfile:  
#     json.dump(dict_params, outfile)


# random.shuffle(files)
sh.rmtree(path_data)
os.makedirs(path_data_train_dogs, mode=0o777)
os.makedirs(path_data_train_cats, mode=0o777)
os.makedirs(path_data_test_dogs, mode=0o777)
os.makedirs(path_data_test_cats, mode=0o777)

# Le compteur s'affichera toutes les "pas_compteur" images traitées
if (n_img <=10):
    pas_compteur = 1
elif (n_img >10) and (n_img<=50):
    pas_compteur = 5
elif (n_img >50) and (n_img<=100):
    pas_compteur = 10
elif (n_img >100) and (n_img<=1000):
    pas_compteur = 100
elif (n_img >1000) and (n_img<=10000):
    pas_compteur = 500
elif (n_img >10000) and (n_img<=50000):
    pas_compteur = 1000
else:
    pas_compteur = 10000

print('dataset_prep.py : Images resizing (height, width) = (', height, ', ', width, ')')
#df_labels = pd.DataFrame()
t_start = time.time()
t_total = 0
for i, elt in enumerate(files):
    file_name = elt.split('/')[-1]
#     if file_name.split('.')[0] == 'dog':
#         df_labels = df_labels.append({'':'1'}, ignore_index=True)
#     elif file_name.split('.')[0] == 'cat':
#         df_labels = df_labels.append({'':'0'}, ignore_index=True)
    # print(file_name, ' : (height, width) = (', height, ', ', width, ')')
    if i%pas_compteur == 0:
        ellapsed = time.time() - t_start
        t_total += ellapsed
        print('dataset_prep.py : number of resized images = ', i, '/', len(files),
              ' in %.2fs ' %ellapsed,
              'for a total of %.2fs.' %t_total)
        t_start = time.time()

    img = cv2.imread(elt)

    #Histogram Equalization
    for j in range(n_channels):
        img[:, :, j] = cv2.equalizeHist(img[:, :, j])
    
    #Image Resizing
    newimg = cv2.resize(img, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    
    cv2.imwrite(path_data + file_name, newimg)

ellapsed = time.time() - t_start
t_total += ellapsed
print('dataset_prep.py : number of resized images = ', i+1, '/', len(files),
              ' in %.2fs ' %ellapsed,
              'for a total of %.2fs.' %t_total)


print('dataset_prep.py : Moved dogs resized images, ',
      n_dog_img_train, ' to the dogs train folder and ',
      n_dog_img_test, 'to the dogs test folder')
#df_labels = pd.DataFrame()
t_start = time.time()
t_total = 0
for i, elt in enumerate(files_dog):
    file_name = elt.split('/')[-1]
    if i < n_dog_img_train:
        sh.move(path_data+file_name, path_data_train_dogs+file_name)
    else:
        sh.move(path_data+file_name, path_data_test_dogs+file_name)
    # print(file_name, ' : (height, width) = (', height, ', ', width, ')')
    if i%pas_compteur == 0:
        ellapsed = time.time() - t_start
        t_total += ellapsed
        print('dataset_prep.py : number of moved dogs images = ', i, '/', len(files),
              ' in %.2fs ' %ellapsed,
              'for a total of %.2fs.' %t_total)
        t_start = time.time()

ellapsed = time.time() - t_start
t_total += ellapsed
print('dataset_prep.py : number of moved dogs images = ', i+1, '/', len(files),
              ' in %.2fs ' %ellapsed,
              'for a total of %.2fs.' %t_total)

print('dataset_prep.py : Moved cats resized images, ',
      n_cat_img_train, ' to the cats train folder and ',
      n_cat_img_test, 'to the cats test folder')
#df_labels = pd.DataFrame()
t_start = time.time()
t_total = 0
for i, elt in enumerate(files_cat):
    file_name = elt.split('/')[-1]
    if i < n_cat_img_train:
        sh.move(path_data+file_name, path_data_train_cats+file_name)
    else:
        sh.move(path_data+file_name, path_data_test_cats+file_name)
    # print(file_name, ' : (height, width) = (', height, ', ', width, ')')
    if i%pas_compteur == 0:
        ellapsed = time.time() - t_start
        t_total += ellapsed
        print('dataset_prep.py : number of moved cats images = ', i, '/', len(files),
              ' in %.2fs ' %ellapsed,
              'for a total of %.2fs.' %t_total)
        t_start = time.time()

ellapsed = time.time() - t_start
t_total += ellapsed
print('dataset_prep.py : number of moved cats images = ', i+1, '/', len(files),
              ' in %.2fs ' %ellapsed,
              'for a total of %.2fs.' %t_total)


print("dataset_prep.py : Number of total images (verify in the config_project.json file : ", n_img)
print("dataset_prep.py : Number of train images (verify in the config_project.json file : ", n_dog_img_train + n_cat_img_train)
print("dataset_prep.py : Number of total images (verify in the config_project.json file : ", n_dog_img_test + n_cat_img_test)

print('dataset_prep.py : End of the images preparation.')

# n_files_cat = 10
# n_files_dog = 10
# 
# cat_files_path = os.path.join(path_orig_img, 'cat.*.jpg')
# dog_files_path = os.path.join(path_orig_img, 'dog.*.jpg')
# 
# cat_files = sorted(glob(cat_files_path))
# dog_files = sorted(glob(dog_files_path))
# 
# for i in range(n_files_cat):
#     sh.copyfile(path_orig_img+prefix_cat+str(i)+img_ext, path_orig_img+str(i)+img_ext)
# 
# for i in range(n_files_dog):
#     sh.copyfile(path_orig_img+prefix_dog+str(i)+img_ext, path_orig_img+str(i+n_files_cat)+img_ext)
    
