# dogs_cats

dogs and cats recognition

# Informations about the images in the datasets

infos_dataset.py: launch this script with a new group of images. The result is a json files (named img_status.json) which gives the min and max of the height and width among all images. This information is important to set the size in the config_project.py script. 

# How to set the initial parameters

config_project.py allows to set the project parameters. Path, images sizes and others important initial
parameters have to be verified (see above step). After the launch of this script, a json file (named config_project.json) is saved in the json folder. It will be load with the launch of the training model script.

# How to prepare the pics

this step is cover by the launch the dataset_prep.py to create pics with specific height and width (depends on the step above), mixed and the labels files named png_labels.csv in the same folder than this pics files.

# How to train the initial model

dl_dogs_cats.py is the script which allows to train an initial model. It begins to load the config.json file.  Then, it trains a model with all images (if b_complete_train = True in config_project.json) and allows to save the model, the history, the partition and the labels data as json files saved in the json folder. This script saved the neurons weights in HD5 format in the json folder. Finally, he accuracies and errors data are displayed in a figure with matplotlib and in real time with the tensorboard feature.

dl_dogs_cats.py is able to to simulate a retrain model if b_complete_train at False in config_project.json. Indeed, instead of taking all images into account, the dataset is segemented in several data set (depending on the value of paramter n_img_package in the config_project.json file).

The CNN model design function (model_design()) is called from prj_functions.py file by dl_dogs_cats.py to before preparing the imageDataGenerator and the batch.

# Important information

The number of images in the dataset (total_img) and the batch_size parameters, should respect the following condition :
	total_img % batch_size = 0
#

If b_complete_train is set at False, it means that the programm is in the retrained mode. So, the number of images in the dataset (total_img) and the n_img_package parameters, should respect the following condition :
	total_img % n_img_package = 0
---> this part of program have to be written 

