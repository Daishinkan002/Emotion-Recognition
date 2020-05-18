import pandas as pd
import numpy as np
import pandas as pd
#from random import shuffle
#from scipy.io import loadmat
#import os
import cv2


    
def get_data(dataset_name='fer2013'):
    dataset_path = 'fer2013/fer2013.csv'
    image_size = (48,48)
    data = pd.read_csv(dataset_path)
    image_set = data['pixels'].tolist()
    w,h = 48,48
    face_data = []
    '''
    for pixel_sequence in image_set:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(w, h)
        face = cv2.resize(face.astype('uint8'), image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions
    
    '''
    for image in image_set:
        face = [int(pixel) for pixel in image.split(' ')]
        face = np.asarray(face).reshape(w,h)
        face = cv2.resize(face.astype('uint8'),image_size)
        face_data.append(face.astype('float32'))
    face_data = np.asarray(face_data)
    face_data = np.expand_dims(face_data, -1)
    emotions = data['emotion'].tolist()
    #print(emotions)
    return face_data,emotions
    
def get_labels():
    return {0:'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


def split_data(x, y, validation_split=.2):
        num_samples = len(x)
        num_train_samples = int(1-validation_split)*num_samples
        train_x = x[:num_train_samples]
        train_y = y[:num_train_samples]
        val_x = x[num_train_samples:]
        val_y = y[num_train_samples:]
        training_data = (train_x,train_y)
        val_data = (val_x,val_y)
        return training_data,val_data



def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model