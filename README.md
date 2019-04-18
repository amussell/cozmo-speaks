# Spring 19 Senior Project
We built a framework for deriving features from images and using those 
features to train a machine learning model (WAC specifically) to map from
those features to a given label (like a color)

We built a dataset and dataset format. You can generate datasets in this same
format using a script we wrote. The format is also outlined in this documentation

## Using the model
Models can be generated using test-vgg-wac.py. This script takes about
3 minutes to run on my laptop. It will output the training and testing
accuracy. It will also give the accuracy for each classifier in the WAC model that is trained. The model is pickled into a file called testModel.pickle

You can load the model using pickle.load(filename). This will instantiate
a LanguageModel object.

## About the model
The model is of type LanguageModel which is a class we wrote. It can easily be used to predict the label of an image by using the predictImageWord function. Please review the doc strings in LanguageModel to review usage.

The model is essentially a pipeline of VGG to WAC. The images are prepared by the model to be passed to VGG to generate a set of features for the given image. Then those features are fed to WAC which produces a guess at the word/label for the image.

It is easy to swap out VGG for another feature generater. You can edit the ImageFeatureGenerator class to add support for doing that.

## Dataset
The dataset is stored as a csv. It contains 4 columns. 

- img (Full image encoded as base64 string)
- label (word label for object in image)
- dimensions (height and width of the full image)
- crop (bounding box of the object in the full image)

The dataset contains images with objects, their bounding boxes, and their labels.
Each image is stored as a base64 encoded string that can be decoded into
a numpy array. The dimensions are stored as a string that represents an array. The easiast way to load and view the dataset is to use the functions
from CS481Dataset.py to load the dataset and decode the images. There is some preprocessing of the pandas Dataframe that is loaded from the csv
to be done before you can use the dataset, and all of that is taken care of if you use the CS481Dataset.py module functions.

## Creating new datasets
You can create new datasets using the cozmo-data-collection.py script.

## Generating New Model From New Dataset
test-vgg-wac.py is used to train, test, and pickle a model using a dataset. You can change the filename that is used to the file path
of whatever dataset you want to train the model on. That dataset must be in the format specified above.

IMPORTANT NOTE 1: As of now the model is trained as if all labels are disjoint. That means that something that is labeled
of one type can only be that type and nothing else. This works so long as your labels are disjoint (i.e. all labels are colors)
However, this is slightly in contrast with the current dataset format which allows multiple labels per image. 

Care is taken in the training of the model in test-vgg-wac.py that when training a classifier for one label that we ensure that all training examples
are accurate. That is to mean, for instance, that positive examples for a red classifier will all contain red example images, and negative
examples will be guarenteed to not be red (Given that the dataset is labeled accurately). However, when evaluating the model after being
trained we evaluate all of the individual label classifiers together. For example if you have objects with the possible classes of 'red',
 'blue', 'square', and 'circle'. For each sample in the dataset the model will try to predict the label for that sample. It is only allowed to 
 pick one label, even though many have multiple labels. 
 
 The way we handle this as of now is to split samples with multiple labels into multiple samples that all have a single label. Foe example if a sample
 in the dataset has an image labeled as a blue square, then this sample would be split into two samples with the same image. One would have the label
 'blue', the other would have the label 'square'. When evaluating, the model is ran on each derivitive sample and tries to guess their label. The model will predict the same thing for each of the derivitive samples because they have the same input image. If an image was marked as a blue square. The model will be ran
 on the image twice. The model will have to choose between blue and square for the label. If it picks blue it will be rewarded for picking blue for
 one of the derivitive samples, but punsihed for picking blue for the derivitive sample that is marked square. This leads to innaccurate 
 evalutation results. 

 If you want accurate results for datasets with labels that are not disjoint, you will need to make changes to test-vgg-wac.py to handle them.

 IMPORTANT NOTE 2: If you are using test-vgg-wac.py on a dataset that has non-disjoint labels (i.e. you have labels for both blue and square: becuase something
  can be both blue and square the labels are not disjoint) then in order for the negative examples to be generated accurately, every sample image must be labeled
  with every label that applies to it. If an image of a blue square is marked in one row of the dataset only as blue and not as a square, then test-vgg-wac.py
  will assume that because it was not labeled as a square it can be used as a negative sample for the 'sqaure' classifier.
  
  ---
# Cozmo Setup
Use this link to setup cozmo: http://cozmosdk.anki.com/docs/initial.html 
- iPhone
    1. Download the cozmo app from the app store.
    2. Connect the charging station to your computer and place Cozmo on it.
    3. Start the app, follow instructions, and enter SDK mode.
- Android
    1. Install adb (android debug bridge)
    2. Enter developer mode on your phone
    3. Plug your phone into your computer
    4. Download the cozmo app to your phone
    5. Start the app, follow instructions, and enter SDK mode.

Use the wifi on your phone to connect to Cozmo. To see Cozmo’s wifi password, lift his arm up and down once while on the charging station.  

### Cozmo SDK
Setup instructions for the SDK and demo scripts are at http://cozmosdk.anki.com/docs/initial.html

----
# Project Setup
1. Install Anaconda https://www.anaconda.com/distribution/
2. Clone the senior design repository.
3. Clone the fastai repo.
    a. `git clone https://github.com/fastai/fastai.git`
    b. `cd fastai/`
4. Create the fastai environment.
    a. With CUDA: `conda env create -f environment.yml`
    b. Or with CPU only: `conda env update -f environment-cpu.yml`
5. In the senior design directory that the scripts are in, create a symbolic link to fastai/old/fastai.
    a. `ln -s ../../fastai/old/fastai/ fastai`
6. Activate the environment (allows you to use the libraries without pip installing all of them).
    a. Linux: `conda activate fastai`
    b. Windows: `activate fastai`
7. Run a Cozmo script
8. There will likely be a number of modules that you will need to install. `pip install <module>` is usually the fix, but you should google the error to find exactly what to do. This project will likely involve a lot of system specific fixes. 

---
# Script Usage

## All Scripts
You can hide error output to console (preferred when not debugging) by adding `2>err.txt` when starting a script. Simply do not put this when you want to see error output.

## Object Detection Script
cd into the directory containing the scripts. With Cozmo and your phone connected to your computer, enter
`python cozmo-object-detection.py 2>err.txt`
This script creates 2 windows: one displays a live feed from Cozmo’s camera; the other displays the processed image with a bounding box drawn around the object being detected.

**Possible Issues:**
- There are paths for the model at the bottom of the script that may need to be adjusted for your directory layout.

## Data Collection Script
cd into the directory containing the scripts. With Cozmo and your phone connected to your computer, enter
`python cozmo-data-collection.py 2>err.txt`
This script will prompt you for the label. After providing the label, 2 windows will pop up. One is the camera feed from Cozmo, the other is the current cropped image you are evaluating. You will be prompted in the console if you want to save the image to the dataset. If you do, enter yes. If you want to see the next image, just press enter with no input. When the script first starts, the first several images will be black and white. Hit enter a few times to flush these out. You will need to rerun the script each time you want to change the label. If you want to change which csv file will be used to store the images, change the variable near the bottom of the script

**Possible Issues:**
- There are paths for the model at the bottom of the script that may need to be adjusted for your directory layout.
- The dataset file specified at the bottom of the script may not exist. Create it and add:
`img,label,dimensions,crop`
- The dataset file might be a different name. Change it in the script to match the actual file.

## Study Script
cd into the directory containing the scripts. With Cozmo and your phone connected to your computer, enter
`python cozmo-study.py 2>err.txt`
This script is for collecting data within the context of a study. It works the same as the data collection script but the prompts are different.

**Possible Issues:**
- There are paths for the model at the bottom of the script that may need to be adjusted for your directory layout.
- The dataset file specified at the bottom of the script may not exist. Create it and add:
`img,label,dimensions,crop`
- The dataset file might be a different name. Change it in the script to match the actual file.

## Color Identification Script
cd into the directory containing the scripts. With Cozmo and your phone connected to your computer, enter
`python cozmo-color-identification.py 2>err.txt`
This script require a model in pickle form to work. This model is loaded in near the bottom of the script. This script is similar to the object detection in the way it runs. It will however prompt you to tell it when to give a prediction. Simply type “y” to get it to predict. It will grab the first 8 frames after signalling that you want to predict. After some time, Cozmo will say his prediction and the prompt will appear again. A prediction is made for each of the 8 frames. The prediction that Cozmo speaks is based on the most frequent output from the 8 images. This may not be the best method (prediction with highest confidence may be a better method). This can be changed in the predict() method. The number of images analyzed during a prediction cycle can be changed in the ImageBuffer variable instantiation near the bottom of the script. 

**Possible Issues:**
- There are paths for the model at the bottom of the script that may need to be adjusted for your directory layout.
- The pickled model may not exist or the path to it may be incorrect.
