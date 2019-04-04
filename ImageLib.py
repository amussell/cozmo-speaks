from typing import Tuple
from BoundingBox import BoundingBox
import math
import numpy as np

epsilon = .00000001

def crop(img: np.array, box: BoundingBox) -> np.array:
    """Crops a numpy array image using the coordinates from a bounding box
    
    Args:
        img (np.array): img to crop
        box (BoundingBox): box to crop
    
    Returns:
        np.array: resulting crop
    """
    return img[int(box.p1.y):int(box.p2.y), int(box.p1.x):int(box.p2.x), :]

def findBestCrop(boundingBox : BoundingBox, imageDimensions: Tuple[int,int], aspectRatio: float) -> BoundingBox:
    """Cwaculates a crop that has the desired aspect ratio and contains the bounding box provided.
    
    Args:
        boundingBox (BoundingBox): the bounding box that needs to be adjusted to the desired aspect ratio
        imageDimensions (Tuple[int,int]): the dimensions (height, width) of the full image that the bounding box is from
        aspectRatio (float): The aspect ratio of the crop that will be calculated and returned (width/height)

    Returns:
        BoundingBox: The smallest bounding box that is as close to the specified aspect ratio as possible
        and contains the specified bounding box (function parameter). The specified bounding box is as 
        centered as possible in the output bounding box.
    """
    if (boxRatioIsEqual(boundingBox, aspectRatio)):
        return boundingBox
    
    newWidth, newHeight, isWider = getSizeFromRatio(boundingBox, aspectRatio)

    if isWider: #Then center horizontaly
        if newWidth <= imageDimensions[1]: # If true, then the new bounding box can be contained within the image
            widthDiff = newWidth - boundingBox.width()
            adjustment = math.floor(widthDiff/2)
            x1 = boundingBox.left() - adjustment
            x2 = x1 + newWidth - 1
            y1 = boundingBox.top()
            y2 = boundingBox.bottom()

            #Check that the new box is within bounds of the image
            if x1 < 0: #if left of the box extends past the left of the image
                x1 = 0
                x2 = x1 + newWidth - 1
            elif x2 >= imageDimensions[1]: #if true
                x2 = imageDimensions[0] - 1
                x1 = x2 - newWidth + 1
            return BoundingBox(x1,y1,x2,y2)
        else:   #The smallest bounding box with the right aspect ratio would exceed the bounds of the image
            #Make the image as wide as possible but still fit in the image
            x1 = imageDimensions[0]
            x2 = imageDimensions[1]
            y1 = boundingBox.top()
            y2 = boundingBox.bottom()
            return BoundingBox(x1,y1,x2,y2)
    else: #The new bounding box is taller than the given bounding box
        if newHeight <= imageDimensions[0]: # The new bounding box can be contained within the image
            heightDiff = newHeight - boundingBox.height()
            adjustment = math.floor(heightDiff/2)
            y1 = boundingBox.top() - adjustment
            y2 = y1 + newHeight - 1
            x1 = boundingBox.left()
            x2 = boundingBox.right()
            
            #Check that the new box is within the bounds of the image
            if y1 < 0: #if true the top of the box extends past the top of the image
                y1 = 0
                y2 = y1 + newHeight - 1
            elif y2 > imageDimensions[1]: #if true the bottom of the box extends past the bottom of the image
                y2 = imageDimensions[0] - 1
                y1 = y2 - newHeight + 1
            return BoundingBox(x1,y1,x2,y2)
        else: #The smallest bounding box with the correct aspect ratio cannot be conatined within the image
            y1 = 0
            y2 = imageDimensions[1]
            x1 = boundingBox.left()
            x2 = boundingBox.right()
            return BoundingBox(x1,y1,x2,y2)



            
        

    
def getSizeFromRatio(boundingBox: BoundingBox, aspectRatio: float) -> Tuple[float,float,bool]:
    """Given a bounding box and desired aspect ratio, this function calculates the closest pixel size (width and height) that
    that is larger than the bounding box and at the correct aspect ratio.

    Returns the width, height, and a boolean value indicating wheter the new aspect ratio is wider or not
    
    Args:
        BoundingBox (BoundingBox): bounding box representing the smallest size
        aspectRatio (float): desired aspect ratio

    Returns:
        Tuple[float,float,bool]: width, height, isWider
    """

    if boundingBox.aspectRatio() > aspectRatio: # The box is wider than we need
        height = boundingBox.width()/aspectRatio
        return boundingBox.width(), height, False
    elif boundingBox.aspectRatio() < aspectRatio: # The box is taller than we need
        width = boundingBox.height()*aspectRatio
        return width, boundingBox.height(), True



def boxRatioIsEqual(box: BoundingBox, aspectRatio: float):
    """Checks if a boxes aspect ratio is equal to the given aspect ratio.
    
    Args:
        box (BoundingBox): box to check aspect ratio for
        aspectRatio (Tuple[int,int]): aspect ratio to check box against (width, height)
    """

    diff = abs(box.aspectRatio() - aspectRatio)
    return diff < epsilon




