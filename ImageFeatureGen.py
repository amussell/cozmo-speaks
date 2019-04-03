import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
import CS481Dataset
import matplotlib.pyplot as plt
import cv2
import numpy as np

class ImageFeatureGen():
    """A model based on vgg19 trained with image net, that generates a 1000 length array of features
    for any image.
    """


    def __init__(self):
        base_model = VGG19(weights = 'imagenet', include_top=True, input_shape=(224, 224, 3))
        self.featureGen = Model(base_model.inputs, base_model.layers[-2].output)
    
    def getFeatures(self, img: np.array) -> np.array:
        """Resizes an image and passes it to vgg19 trained on imagenet. Uses the second to last
        layer to generate a 1000 length feature array and returns that array.
        
        Args:
            img (np.array): a 3 channel rgb image
        
        Returns:
            np.array: feature array with 1000 features.
        """

        prePropImg = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        prePropImg = np.expand_dims(prePropImg, axis=0)
        fs = self.featureGen.predict(prePropImg)
        fs = np.squeeze(fs)
        return fs