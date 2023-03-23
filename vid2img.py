import cv2
import os


class VideoToImage:

    def __init__(self, videoPath, savePath):
        self.videoPath = videoPath
        self.savePath = savePath
        self.vidCap = cv2.VideoCapture(videoPath)        
        self.count = 0
        self.recordFrame  = 0
        
        success, image =self.vidCap.read()
        self.success = success
        self.image = image
        self.imagePathArray = []
        
    def variance_of_laplacian(self):
        """
        compute the Laplacian of the image and return the focus measure
        """
        return cv2.Laplacian(self.image, cv2.CV_64F).var()

    def saveImage(self):

        while self.success:

            if self.recordFrame == self.count:
                image_path = os.path.join( self.savePath, "frame%d.png" % self.count )                
                cv2.imwrite(image_path, self.image)
                self.imagePathArray.append(image_path)

            self.success, self.image = self.vidCap.read()
            
            if self.count == self.recordFrame :
                self.recordFrame += 10
            
            self.count += 1
            
        return self.imagePathArray
