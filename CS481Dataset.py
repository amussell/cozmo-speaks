import pandas as pd 
import numpy as np
import PIL
from PIL import Image
import base64
from ast import literal_eval

def encodeAndSaveImage(df: pd.DataFrame, img: PIL.Image, label: str, boundingBox):
    """
     Saves and image to a cs481 dataset dataframe. 
     The dataframe has 3 columns in this order [img: base64 encoded string, label: string, image dimensions: list]

    The image is converted to a numpy array and then base64 encoded.

    The dimensions are stored as [height, width, channels]
    This function expects all images to have 3 channels and does not check if they do, all images are saved indicating 3 channels.
    Behavior is undefined if an image does not have exactly 3 channels.

    Args:
        df (pd.DataFrame): the dataframe to append the labeled image to
        img (PIL.Image): the image to store in the dataset
        label (str): the label to be stored with the image
        boundingBox: array of coordinates for cropping image
    """
    
    imgNumpy = np.asarray(img)
    imgStr = base64.b64encode(imgNumpy)

    #Couple of things going on below. PIL images store size as (width, height)
    # We reverse this to be (height, width) and then append 3 to get 
    # [height, width, channels] in a list. Also the string that is encoded from
    # b643encode is formated "b'[A-Za-z0-9_-]*'" in order to remove the first 2
    # charachters (b') and the last charachter (') we trim them off by indexing
    # from [2:-1]. 
    df.loc[len(df)] = [str(imgStr)[2:-1], label, list(img.size)[::-1] + [3], boundingBox]

def decodeImageFromRow(df: pd.DataFrame, row: int):
    """ Decodes a base64 encoded string from the dataset into a numpy array.
    
    Args:
        df (pd.DataFrame): the dataframe containing the dataset
        row (int): the row to read the image from
    
    Returns:
        np.array: the image being decoded from the given row in a numpy array
    """

    buffer = base64.b64decode(df.img.iloc[row])
    img = np.frombuffer(buffer, dtype= np.uint8).reshape(tuple(df.dimensions.iloc[row]))
    return img

def loadFromCSV(filename: str):
    """Loads the dataset from a csv and cleans the dimensions column so that
    it is not a string, but instead a list.
    
    Args:
        filename (str): the filename of the csv containing the dataset
    """

    data = pd.read_csv(filename)

    # When a column containing a list is written to a csv and read back in the column becomes a string.
    # this convers the string representation of a list back to a list.
    data.dimensions = data.dimensions.apply(lambda l : literal_eval(l)) 
    return data