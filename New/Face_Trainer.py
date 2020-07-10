#Program to train with the faces and create a YAML file

import cv2 
import numpy as np 
import os 
from PIL import Image 
def face_trainer():
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        Face_ID = 0 
        pev_person_name = "Manuel"
        y_ID = []
        x_train = []

        Face_Images = os.path.join(os.getcwd(), "Face_Images")  
        print (Face_Images)

        for root, dirs, files in os.walk(Face_Images): 
                for file in files:  
                        if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):  
                                path = os.path.join(root, file)
                                person_name = os.path.basename(root)
                                print(path, person_name)

			
                                if pev_person_name!=person_name:  
                                        Face_ID=Face_ID+1 
                                        pev_person_name = person_name

			
                                Gery_Image = Image.open(path).convert("L") 
                                Crop_Image = Gery_Image.resize( (550,550) , Image.ANTIALIAS) 
                                Final_Image = np.array(Crop_Image, "uint8")
                                #print(Numpy_Image)
                                faces = face_cascade.detectMultiScale(Final_Image, scaleFactor=1.2,minNeighbors=5) 
                                print (Face_ID,faces)

                                for (x,y,w,h) in faces:
                                        roi = Final_Image[y:y+h, x:x+w] 
                                        x_train.append(roi)
                                        y_ID.append(Face_ID)

        recognizer.train(x_train, np.array(y_ID)) 
        recognizer.write("face-trainner.yml") 
