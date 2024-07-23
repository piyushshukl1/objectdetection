# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 19:13:22 2022

@author: wave
"""

import cv2
import numpy as np
with_mask = np.load('with_mask.npy')
without_mask = np.load('with_outmask.npy')
without_mask.shape
with_mask = with_mask.reshape(200,50*50*3)
without_mask = without_mask.reshape(200,50*50*3)
with_mask.shape
without_mask.shape
X =np.r_[with_mask,without_mask]
X.shape
labels = np.zeros(X.shape[0])
labels[200:] = 1.0
names ={0 : 'Mask', 1 : 'No Mask'}
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train, x_test ,y_train, y_test = train_test_split(X,labels, test_size=0.25)
x_train.shape
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
x_train =pca.fit_transform(x_train)
x_train[0]
x_train.shape
svm = SVC()
svm.fit(x_train, y_train)
x_test = pca.transform(x_test)
y_pred =svm.predict(x_test)
accuracy_score(y_test, y_pred)
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
            face = face.reshape(1,-1)
            face = pca.transform(face)
            pred = svm.predict(face)
            n = names[int(pred)]
            cv2.putText(n)
            print(n)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27:
            
            break 
            
capture.release()
cv2.destroyAllWindows()
