

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:19:12 2022

@author: wave
"""

import cv2
import numpy as np
haar_data = cv2.CascadeClassifier('C:/Users/wave/Desktop/facemask/haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
data = []
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))
            print(len(data))
            if len(data)<400:
                data.append(face)
        cv2.imshow('result',img)
        np.save('with_outmask.npy', data)
        if cv2.waitKey(2) == 27 or len(data) >=200:
            
            break 
            
capture.release()
cv2.destroyAllWindows()
import matplotlib.pyplot as plt
plt.imshow(data[6])
