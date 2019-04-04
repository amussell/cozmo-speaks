import pandas as pd 
import numpy as np
import PIL
from PIL import Image
import base64
from ast import literal_eval
from typing import List, Tuple
import ImageLib
from BoundingBox import BoundingBox

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

def decodeImgString(imgStr: str, dimensions: List) -> np.array:
    """Decodes a base64 string into a numpy array.
    
    Args:
        imgStr (str): base64 encoded numpy array image
    """
    buffer = base64.b64decode(imgStr)
    img = np.frombuffer(buffer, dtype= np.uint8).reshape(dimensions)
    return img

def getImgCrop(img: np.array, crop: Tuple[int,int,int,int], aspectRatio: float) -> np.array:
    """Uses the crop specified by the dataset to calculate a crop that is of the correct aspect ratio
    then crops the image from the dataset and returns the resulting cropped image.

    It is possible the resulting crop will not have the specified aspect ratio. In that case it will be the
    the closest aspect ratio that could be achieved while keeping the crop within the bounds of the image. (Otherwise
    padding the crop with 0s or other values would be required)
    
    Args:
        img (np.array): image from dataset
        crop (Tuple[int,int,int,int]): location of object in image (y, x, height, width)
        aspectRatio (float): desired aspect ratio
    
    Returns:
        np.array: [description]
    """
    x1, y1, x2, y2 = crop
    box = BoundingBox(x1, y1, x2, y2)
    bestCrop = ImageLib.findBestCrop(box, (img.shape[0], img.shape[1]), aspectRatio)
    cropImg = ImageLib.crop(img, bestCrop)
    return cropImg
    
def adjustCrop(crop: Tuple[int,int,int,int], dimensions: Tuple[int,int,int]):
    """Corrects crop coordinates to be withing bounds of dimensions
    
    Args:
        crop (Tuple[int,int,int,int]): (x1, y1, x2, y2)
        dimensions (Tuple[int,int,int]): (height, width, channels)
    """
    x1, y1, x2, y2 = crop
    if x1 < 0: x1 = 0
    if x1 > dimensions[1]: x1 = dimensions[1] - 1
    if y1 < 0: y1 = 0
    if y1 > dimensions[0]: y1 = dimensions[0] - 1
    if x2 < 0: x2 = 0
    if x2 > dimensions[1]: x2 = dimensions[1] - 1
    if y2 < 0: y2 = 0
    if y2 > dimensions[0]: y2 = dimensions[0] - 1
    return x1, y1, x2, y2

def genCropsWithAspectRatio(data: pd.DataFrame, aspectRatio: float)-> pd.DataFrame:
    """Creates a column in the dataframe with cropped images at the specified aspect ratio.
    Uses the crop column in the dataframe to calculate a new crop with the specified aspect ratio.

    Not all images are gauranteed to have the specified aspect ratio. 
    
    Args:
        data (pd.DataFrame): data needs these columns [img: (np.array), crop: (tuple in form x1,y1,x2,y2)]
        aspectRatio (float): the aspect ratio that all cropped images need to be.
    
    Returns:
        pd.DataFrame: dataframe with the new column of cropped images (column is named cropImg)
    """
    data['cropImg'] = data.apply(lambda row: getImgCrop(row.img, row.crop, aspectRatio), axis=1)
    return data


def loadFromCSV(filename: str):
    """Loads the dataset from a csv and cleans the dimensions column so that
    it is not a string, but instead a list. Also decodes base64 encoded images
    into np.arrays. Also cleans crop tuples to be within dimensions of images and
    to be tuples rather than strings. Also lower cases labels and splits them into
    lists of labels.
    
    Args:
        filename (str): the filename of the csv containing the dataset
    """

    data = pd.read_csv(filename)

    # When a column containing a list is written to a csv and read back in the column becomes a string.
    # this convers the string representation of a list back to a list.
    data.dimensions = data.dimensions.apply(lambda l : literal_eval(l)) 
    data.crop = data.crop.apply(lambda t : literal_eval(t))
    data.img = data.apply(lambda row : decodeImgString(row.img, row.dimensions), axis=1)
    data.crop = data.crop.apply(lambda c : tuple(map(round, c)))
    data.crop = data.apply(lambda row : adjustCrop(row.crop, row.dimensions), axis=1) # some crops have negative indeces. We need to adjust all crops to be within bounds of image
    data.label = data.label.apply(lambda s : s.lower().split())
    return data