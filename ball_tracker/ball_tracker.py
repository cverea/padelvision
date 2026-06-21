import argparse
from .models import TrackNet
import cv2
import numpy as np
import pickle
import pandas as pd 

class BallTracker:
    def __init__(self, input_video_path, save_weights_path, n_classes):
        self.input_video_path = input_video_path
        self.save_weights_path = save_weights_path
        self.n_classes = n_classes
        self.width, self.height = 640, 360
        self.model = self.load_model()
        self.ball_coordinates = {}
        
    # Load the trained model
    def load_model(self):
        modelFN = TrackNet
        model = modelFN(self.n_classes, input_height=self.height, input_width=self.width)
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        model.load_weights(self.save_weights_path)
        return model
    
    def predict(self, read_from_stub=False, stub_path=None):
        # Read the ball detections from a precomputed stub file
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections        

        # Read the video and process the frames
        video = cv2.VideoCapture(self.input_video_path)        
        output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))        
        currentFrame = 0
        img, img1, img2 = None, None, None
         
                       
        # Capture the first frame
        video.set(1, currentFrame)
        ret, img1 = video.read()
        currentFrame += 1
        img1 = cv2.resize(img1, (self.width, self.height))
        img1 = img1.astype(np.float32)
        
        # Capture the second frame
        video.set(1, currentFrame)
        ret, img = video.read()
        currentFrame += 1
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype(np.float32)
        
        while True:
            img2 = img1
            img1 = img
            
            video.set(1, currentFrame)
            ret, img = video.read()
            
            if not ret:
                break
            
            
            img = cv2.resize(img, (self.width, self.height))
            img = img.astype(np.float32)
            
            # Concatenate the three frames
            X = np.concatenate((img, img1, img2), axis=2)
            X = np.rollaxis(X, 2, 0)
            pr = self.model.predict(np.array([X]))[0]
            pr = pr.reshape((self.height, self.width, self.n_classes)).argmax(axis=2)
            pr = pr.astype(np.uint8)
            # Resize the heatmap to the output dimensions
            heatmap = cv2.resize(pr, (output_width, output_height))
            ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
            
            # Detect the ball using the HoughCircles method
            circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
            
            # Store the ball coordinates
            if circles is not None and len(circles) == 1:
                x = int(circles[0][0][0])
                y = int(circles[0][0][1])
                print(currentFrame, x, y)
                self.ball_coordinates[currentFrame] = (x, y)                                        
            else:
                self.ball_coordinates[currentFrame] = None
                
            currentFrame += 1
        
        video.release()
        print("Ball coordinates:", self.ball_coordinates)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(self.ball_coordinates, f)

        return self.ball_coordinates   
    
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_path", type=str)
    parser.add_argument("--save_weights_path", type=str)
    parser.add_argument("--n_classes", type=int)
    
    args = parser.parse_args()
    tracker = BallTracker(args.input_video_path, args.save_weights_path, args.n_classes)
    tracker.process_video()
