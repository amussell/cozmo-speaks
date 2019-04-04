from shapely.geometry import Point

class BoundingBox():

    def __init__(self, x1, y1, x2, y2):
        self.p1 = Point(x1,y1)
        self.p2 = Point(x2,y2)
    
    def top(self):
        """ Finds the lowest y value for this bounding box. 

        Note: Image coordinates are inverted so a lower y coordinate is higher in the image.
        """
        return min(self.p1.y, self.p2.y)
    
    def bottom(self):
        """Finds the highest y value for this bounding box
        """
        return max(self.p1.y, self.p2.y)
    
    def left(self):
        """ Finds the lowest x value for this box
        """
        return min(self.p1.x, self.p2.x)

    def right(self):
        """Finds the highest x value for this box
        """
        return max(self.p1.x, self.p2.x)
    
    def width(self):
        """Gets the width of this box
        """
        return abs(self.p1.x - self.p2.x) + 1
    
    def height(self):
        """Gets the height of this box
        """
        return abs(self.p1.y - self.p2.y) + 1


    def aspectRatio(self):
        """Gets the aspect ratio (width/height) of this box
        """
        return self.width()/self.height()



