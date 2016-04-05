#!/usr/bin/env python

import os.path
import numpy as np
import cv2

def preparePhoto(file_path):
  face_cascade = cv2.CascadeClassifier('/home/patryk/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
  img = cv2.imread(file_path);
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
  i = 0
  for (x, y, w, h) in faces:
    cropped_img = img[y:y+h, x:x+w]
    scaled_img = cv2.resize(cropped_img, (400, 400))
    cv2.imwrite(file_path, scaled_img)
    print file_path + " saved!"
    i+=1
    

if __name__ == "__main__":
  FOLDER_NAME = "photos"
  SEPARATOR = ";"
 
  label = 0
  csv_file = open('data.txt', 'w')

  for dirname, dirnames, filenames in os.walk(FOLDER_NAME):
    for subdirname in dirnames:
      subject_path = os.path.join(dirname, subdirname)
      for filename in os.listdir(subject_path):
        abs_path = "%s/%s" % (subject_path, filename)
        csv_file.write("%s%s%d\n" % (abs_path, SEPARATOR, label))
        preparePhoto(abs_path)
      label = label + 1
  csv_file.close()

