#!/usr/bin/python3
# -*- coding: utf8 -*-

# With a new dataset, this script allows to inform about the size min and max
# and about the number of images 

#import shutil as sh
import os
import pandas as pd
import cv2  # installation : opencv
import json
import time

from glob import glob

print("infos_dataset.py : Download init parameters from json file")
path_project = os.path.dirname(os.path.abspath(__file__))
path_json = path_project+"/json/"

with open(path_json+'config_project.json', 'r') as json_file:  
    dict_params = json.load(json_file)

print("infos_dataset.py : set variables")
# Number of dog and cat images trained at the first launch
n_img_package = dict_params['n_img_package']
# Directory where the images are saved
path_orig_img = dict_params['path']['orig_img']
# Image status json file name to save
file_img_status_json = dict_params['file_name']['img_status']
img_ext = "."+dict_params['img']['extension']

prefix_dog = "dog."
prefix_cat = "cat."

files_dog_path = os.path.join(path_orig_img, prefix_dog+'*'+img_ext)
files_cat_path = os.path.join(path_orig_img, prefix_cat+'*'+img_ext)
files_dog = sorted(glob(files_dog_path))
files_cat = sorted(glob(files_cat_path))
n_dog_img = len(files_dog)
n_cat_img = len(files_cat)
files = []
files.extend(files_dog)
files.extend(files_cat)
n_img = n_dog_img + n_cat_img


height_max = 0
height_min = 5000
width_max = 0
width_min = 5000
lst_channel = [3]


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

df_labels = pd.DataFrame()
t_start = time.time()
t_total = 0
for i, elt in enumerate(files):
    file_name = elt.split('/')[-1]
    if i%pas_compteur == 0:
        ellapsed = time.time() - t_start
        t_total += ellapsed
        print('infos_dataset.py : number of images parsed = ', i+1, '/', len(files),
              ' in %.2fs ' %ellapsed,
              'for a total of %.2fs.' %t_total)
        t_start = time.time()
        
    img = cv2.imread(path_orig_img + file_name)
    height_max = max(height_max, img.shape[0])
    height_min = min(height_min, img.shape[0])
    width_max = max(width_max, img.shape[1])
    width_min = min(width_min, img.shape[1])
    if img.shape[2] != 3:
        lst_channel.append(img.shape[2])
    else:
        pass

ellapsed = time.time() - t_start
t_total += ellapsed
print('infos_dataset.py : number of images parsed = ', i+1, '/', len(files),
      ' in %.2fs ' %ellapsed,
      'for a total of %.2fs.' %t_total)


img_status = {'n_img':n_img,
              'n_dog_img':n_dog_img,
              'n_cat_img':n_cat_img,
              'height_max':height_max,
              'height_min':height_min,
              'width_max':width_max,
              'width_min':width_min,
              'lst_channel':lst_channel}

with open(path_json+file_img_status_json, 'w') as outfile:  
    json.dump(img_status, outfile)

