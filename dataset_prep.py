#!/usr/bin/python3
# -*- coding: utf8 -*-

# labels : 0 for a cat, and 1 for a dog

import shutil as sh
import os
import random
import csv
import numpy as np
import pandas as pd
import cv2  # installation : opencv

from glob import glob
from builtins import enumerate

path_data = "/home/tostakit/Téléchargements/dogsvscats/"
path_sample = "/home/tostakit/Téléchargements/dogsvscats/sample/"
size_width = 150
size_length = size_width
prefix_dog = "dog."
img_ext = ".jpg"

files_path = os.path.join(path_data, '*.jpg')
files = sorted(glob(files_path))
random.shuffle(files)
len(files)

df_labels = pd.DataFrame()
for i, elt in enumerate(files):
    file_name = elt.split('/')[-1]
    if file_name.split('.')[0] == 'dog':
        df_labels = df_labels.append({'':'1'}, ignore_index=True)
    else:
        df_labels = df_labels.append({'':'0'}, ignore_index=True)
    print(file_name)

    img = cv2.imread(path_data + file_name)
    newimg = cv2.resize(img, dsize=(size_width, size_length), interpolation=cv2.INTER_CUBIC)
    
    cv2.imwrite(path_sample + str(i) + img_ext, newimg)
    # sh.copyfile(path_data+file_name, path_sample+str(i)+img_ext)

df_labels.to_csv(path_sample + 'png_labels.csv', sep=',', index=False, header=None)
# df = pd.read_csv(path_sample+'png_labels.csv', header=None)

# n_files_cat = 10
# n_files_dog = 10
# 
# cat_files_path = os.path.join(path_data, 'cat.*.jpg')
# dog_files_path = os.path.join(path_data, 'dog.*.jpg')
# 
# cat_files = sorted(glob(cat_files_path))
# dog_files = sorted(glob(dog_files_path))
# 
# for i in range(n_files_cat):
#     sh.copyfile(path_data+prefix_cat+str(i)+img_ext, path_data+str(i)+img_ext)
# 
# for i in range(n_files_dog):
#     sh.copyfile(path_data+prefix_dog+str(i)+img_ext, path_data+str(i+n_files_cat)+img_ext)
    
