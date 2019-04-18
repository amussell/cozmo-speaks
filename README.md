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