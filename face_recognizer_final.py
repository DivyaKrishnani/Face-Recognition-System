import cv2
import os
import numpy as np
from PIL import Image

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Using LBPH face recognizing algorithm 
#LBPH works for images of different sizes too
recognizer= cv2.createLBPHFaceRecognizer()

def images_and_labels_list(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    
    images=[]   #Face images will be present
    labels=[]   #Labels assigned to those faces will be present
    
    for image_path in image_paths:
        image_conv = Image.open(image_path).convert('L')   #Coverts to grayscale
        image = np.array(image_conv, 'uint8')              #Converts to numpy array since furthur functions use this array to work
        label= int(os.path.split(image_path)[1].split(".")[0].replace("subject",""))    #Getting Label
        #Detecting face in the image using OpenCV
        faces = faceCascade.detectMultiScale(image,1.1)    
        
        for(x,y,w,h) in faces:
            images.append(image[y: y+h, x: x+w])
            labels.append(label)
            res=cv2.resize(image[y: y+h, x: x+w],None,fx=4, fy=4,interpolation = cv2.INTER_LINEAR)
            cv2.imshow("faces being added to training set",res)
            cv2.waitKey(200)

    return images, labels    #Returning images list and labels list

path='Known_faces'  #Path to the Dataset

images, labels = images_and_labels_list(path) #Calling images_and_labels_list and get the face images and the corresponding labels
cv2.destroyAllWindows()

recognizer.train(images, np.array(labels))  #Perform the tranining 

path_to_test='Unknown_faces'   #Path to the Dataset

image_paths=[os.path.join(path_to_test,f) for f in os.listdir(path_to_test)]   # Appending the images into image_paths

for image_path in image_paths:
    predict_image_conv = Image.open(image_path).convert('L') #Coverts to grayscale
    predict_image = np.array(predict_image_conv, 'uint8')  #Makes an array of pixels
    faces = faceCascade.detectMultiScale(predict_image)   #Detects only the face in the picture and make an array

    for(x,y,w,h) in faces:
        label_predicted, confidence = recognizer.predict(predict_image[y: y+h, x: x+w])   #Returns outcome and the confidence of the outcome
        label_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject",""))
   
        if label_predicted == label_actual:
            print "This image is of {}".format(label_predicted) 
        else:
            print "This image not recognized"
        res=cv2.resize(predict_image[y: y+h, x: x+w],None,fx=4, fy=4,interpolation = cv2.INTER_LINEAR)
        cv2.imshow("Recognizing faces",res)   #Shows the face currently recognizing
        cv2.waitKey(1000)
               

