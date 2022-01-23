from __future__ import print_function
import cv2
import argparse
import numpy as np

def count_face(image_name):
    img = cv2.imread(image_name)

    face_cascade = cv2.CascadeClassifier()
    if not face_cascade.load(cv2.samples.findFile('data/haarcascades/haarcascade_frontalface_alt.xml')):
        print("Error loading face cascade file")
        exit(0)

    
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    faces = face_cascade.detectMultiScale(gray_img)
    #cv2.imshow("Gray",gray_img)
    #cv2.waitKey(0)
    if len(faces)>0:
        print("\n\n.......Image Approved for testing......\n")
        #print(len(faces))
        return len(faces)
        
    else:
        print("\n\n........No Face Found...............")
        return 0
    

#count_face()