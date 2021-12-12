import cv2 as cv
import numpy as np
import keras.utils
import os
import PIL
import tensorflow as tf
from tensorflow import keras

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
class ProcessVideo:
    """
    Process the viedo and classifies every frame
    Keeps track of number of object for each class
    Saves the first frame for each object
    """
    def __init__(self, model_name = 'my_model.h5', model_img_height = 256, model_img_width = 256, video_file = 'demo.mkv', save_frames_dir = "SavedFrames"):
        self.model = keras.models.load_model(model_name)
        self.img_height = model_img_height
        self.img_width = model_img_width
        self.class_names = ["bus", "car", "truck"]
        self.class_count = {"bus": 0, "car": 0, "truck": 0}
        self.isVehicleChanged = False
        self.lastVehicleType = ""
        self.listOfPredications = []
        self.cap = cv.VideoCapture(video_file)
        self.SavedFramesDir = save_frames_dir
        self.listOfPredications = []
    
    #check if a new object should be added
    def CheckToAdd(self):
        """
        Check if an object should be added or not, and return the object type
        """
        #The video is 30fps, so each second has 30 images
        #checking batch of images instead of just one, helps in avoiding wrong predications in a few frames from counting as a new object
        if(len(self.listOfPredications) < 10):
            return False, ""
        else:
            
            #finds the most comman vehicle in the last 10 predications
            counter = 0
            vehicleType = self.listOfPredications[0]
            
            for i in self.listOfPredications:
                curr_frequency = self.listOfPredications.count(i)
                if(curr_frequency> counter):
                    counter = curr_frequency
                    vehicleType = i
            
            #check if the current vehile is the same as the last one, if yes then it's the same object, otherwise it's considered as a new object
            if(self.lastVehicleType != vehicleType):
                self.lastVehicleType = vehicleType
                if(self.listOfPredications.count(vehicleType) < len(self.listOfPredications) * 0.7):
                    self.listOfPredications.clear()
                    return False, vehicleType
                else:
                    self.listOfPredications.clear()
                    return True, vehicleType
            else:
                self.listOfPredications.clear()
                return False, vehicleType
    
    def AddObject(self, frame):
        """
        Adds an object
        """
        shouldAdd, vehicleType = self.CheckToAdd()
        if(shouldAdd):
            self.class_count[vehicleType]+=1
            
            # save the first frame of each object in separate folder according to class
            im = PIL.Image.fromarray(frame)
            im.save(os.path.join(self.SavedFramesDir, vehicleType, f"{self.class_count[vehicleType]}.jpg"))
        else:
            return
         
    
    
    def PlayVideo(self):
        """
        Plays the video and classifiy each frame
        """
        if (self.cap.isOpened() == False):
             print("Error opening video stream or file")
        
        # # Read until video is completed
        while (self.cap.isOpened()):
             # Capture frame-by-frame
             ret, frame = self.cap.read()
             if ret == True:
                 #resize the image to fit the model
                 frameToProcess = (np.expand_dims(frame,0))
                 frameToProcess = tf.image.resize(frameToProcess,
                                 (self.img_height, self.img_width))
                 result = self.model.predict(frameToProcess)
                 
                 #append to the predications list
                 self.listOfPredications.append(self.class_names[np.argmax(result)])
                 
                 
                 self.AddObject(frame)
                 
                 font = cv.FONT_HERSHEY_SIMPLEX
                 #Show the predication for current image
                 frame = cv.putText(frame, self.class_names[np.argmax(result)] ,(50, 50), font, 2,(0,255,0), 2, cv.LINE_AA)
                 
                 #Show the object counts for each class
                 frame = cv.putText(frame, f"Bus Count: {self.class_count['bus']}" ,(50, 100), font, 2,(0,255,0), 2, cv.LINE_AA)
                 frame = cv.putText(frame, f"Car Count: {self.class_count['car']}" ,(50, 150), font, 2,(0,255,0), 2, cv.LINE_AA)
                 frame = cv.putText(frame, f"Truck Count: {self.class_count['truck']}" ,(50, 200), font, 2,(0,255,0), 2, cv.LINE_AA)
                 cv.imshow('Frame', frame)
        
             # Press Q on keyboard to  exit
                 if cv.waitKey(25) & 0xFF == ord('q'):
                     break
        
             # Break the loop
             else:
                 break
        # When everything done, release the video capture object
        self.cap.release()
        
        # # Closes all the frames
        cv.destroyAllWindows()
        
        print(self.get_objects_count())
        
        
    def get_objects_count(self):
        return self.class_count
        
test = ProcessVideo()
test.PlayVideo()