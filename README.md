# Spring 19 Senior Project
We built a framework for deriving features from images and using those 
features to train a machine learning model (WAC specifically) to map from
those features to a given label (like a color)

We built a dataset and dataset format. You can generate datasets in this same
format using a script we wrote. The format is also outlined in this documentation

## Using the model
Models can be generated using test-vgg-wac.py. This script takes about
3 minutes to run on my laptop. It will output the training and testing
acuracy. It will also give the accuracy for each classifier in the WAC model that is trained. The model is pickled into a file called testModel.pickle

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