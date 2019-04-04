from ImageFeatureGen import ImageFeatureGen
from wac import WAC
import operator
import numpy as np
from sklearn import linear_model
import pandas as pd
import CS481Dataset
from tqdm import tqdm
import pickle

class LanguageModel():
    """Model that applies wac to images using VGG19 as a feature generator for the images.
    Given a dataframe with the headings (word, img, annotation) where word is the word is a string
    the img is an np.array, and a the annotation is a 0 or 1 indicating whether or not the image
    is classified by that word.
    """

    
    def __init__(self):
        self.imgFeatureGen = ImageFeatureGen()
        lr_spec=(linear_model.LogisticRegression,{'penalty':'l1'})
        self.wac = WAC('cozmo', lr_spec)

    def addImages(self, imgData: pd.DataFrame) -> None:
        """Adds image feature data to the wac model. The image feature data is a dataframe with the headings (word, imgFeatures, annotation)
         where word is the word is a string the imgFeatures is an np.array, and a the annotation is a 0 or 1 indicating whether or not the image 
         is classified by that word.
        
        Args:
            imgData (pd.DataFrame): dataframe with the headings (word, imgFeatures, annotation)
        """

        for word in set(imgData.word):
            wordData = imgData[imgData.word == word]
            posWordData = wordData[wordData.annotation == 1]
            negWordData = wordData[wordData.annotation == 0]

            self.wac.add_multiple_observations(word, posWordData.imgFeatures.values.tolist(), [1] * len(posWordData))
            self.wac.add_multiple_observations(word, negWordData.imgFeatures.values.tolist(), [0] * len(negWordData))

    def train(self) -> None:
        """Trains the wac models on the image data that has been added using addImages()
        """

        self.wac.train()

    def predictImageWord(self, img: np.array, imgFeatures=None) -> str:
        """Given an image of any size, the most probable word label is returnded.

        The img past in is resized, passed to a feature generator (based on vgg19) 
        and then the features are fed to WAC. The word with the highest probability
        returned from WAC is returned.
        
        Args:
            img (np.array): 3 channel image
            imgFeatures (np.array) : imageFeature array. use this to avoid overhead of recalculating imgFeatures from image.
        
        Returns:
            str: word label
        """

        mostProbWord = None
        if(imgFeatures is not None):
            mostProbWord = self.wac.predictWord(imgFeatures)
        else:
            imgFeaturesCalc = self.imgFeatureGen.getFeatures(img)
            mostProbWord = self.wac.predictWord(imgFeaturesCalc)
        return mostProbWord
    
    def savePickle(self, filename):
        data = {'wac' : self.wac}
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle)

    def loadPickle(self, filename):
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
            self.wac = data['wac']
    
    

def prepareDataset(df: pd.DataFrame) -> pd.DataFrame:
    """Takes an input dataframe formated with headers (img, labels, dimensions) [This is the CS481Dataset format] where
    each row contains a base64 encoded image string, a label for that image and the dimenions of that image.
    It then adds annotations for each image (0 or 1) indicating whether or not the label applies to the image (1 means the label applies)

    It also decodes each image into a numpy.array

    Then it creates an equal number of negative examples for each image label. It outputs a dataframe with headers (img, label, annotation)
    
    Args:
        df (pd.DataFrame): dataframe (should be created using CS481Dataset.loadFromCSV())
    
    Returns:
        pd.DataFrame: [description]
    """

    data = CS481Dataset.genCropsWithAspectRatio(df, 1)

    featureGen = ImageFeatureGen()
    data['imgFeatures'] = [featureGen.getFeatures(img) for img in data.cropImg]

    newData = pd.DataFrame(columns=data.columns)
    for index, row in df.iterrows():
        for label in row.label:
            newRow = row.copy()
            newRow.label = label
            newData = newData.append(newRow)

    newData['annotation'] = 1
    
    sampleDF = newData.copy() # Create a dataframe to sample from so that we can append to the original df without contaminating it for sampling

    for label in set(newData.label):
        numPos = len(sampleDF[sampleDF.label == label])
        negDf = data[data.label.apply(lambda l : label not in l)]
        negSamples = negDf.sample(n=numPos)
        negSamples['annotation'] = 0
        negSamples['label'] = label
        newData = newData.append(negSamples, ignore_index=True)

    newData = newData.drop(['dimensions'], axis=1)
    newData = newData.rename(columns={'label' : 'word'})
    return newData

def loadLanguageModelFromDataset(filename: str):
    langMod = LanguageModel()
    data = prepareDataset(CS481Dataset.loadFromCSV(filename))
    print('Adding images to WAC')
    langMod.addImages(data)
    print('Training WAC')
    langMod.train()
    return langMod
