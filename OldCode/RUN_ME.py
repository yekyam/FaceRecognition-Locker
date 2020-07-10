import cv2
import numpy as np
import os
from PIL import Image

labels = ["Manuel"]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.createLBPHFaceRecognizer()
#recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load("face-trainner.yml")

cap = cv2.VideoCapture(0) #might need to change to pi cam cause I don't have a USB camera

while(True):
    
    ret, img = cap.read()
    gray =cv2cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
    
    id_, conf = recognizer.predict(roi_gray)
    
    if conf>=80:
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labaes[id_]
        cv2.putText(img, name, (x,y), font, 1, (0,0,255), 2)
    
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('Preview',img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
