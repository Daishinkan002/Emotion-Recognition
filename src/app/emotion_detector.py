from statistics import mode
import cv2
from keras.models import load_model
import numpy as np
from dataset import get_labels, load_detection_model
from Image_Updation import apply_offsets, draw_bounding_box, draw_text
from data_preprocessing import preprocess_input







detection_model_path = 'data/haarcascades/haarcascade_frontalface_alt.xml'
emotion_model_path = 'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels()

frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
face_cascade = cv2.CascadeClassifier()
face_cascade = cv2.CascadeClassifier()

if not face_cascade.load(cv2.samples.findFile('data/haarcascades/haarcascade_frontalface_alt.xml')):
    print("Error loading face cascade file")
    exit(0)
emotion_target_size = emotion_classifier.input_shape[1:3]


emotion_window = []
#image_name = 'index.jpeg'

def predict(image_name):
    img = cv2.imread(image_name)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)
    faces = face_cascade.detectMultiScale(gray_image)
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            sys.exit("\nImage is too large for predicting\n\n")
            
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        #print(emotion_prediction)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        #print("Emotion_label_arg : ",emotion_label_arg)
        emotion_text = emotion_labels[emotion_label_arg]
        print("Emotion  : ",emotion_text)
        emotion_window.append(emotion_text)
        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))
        color = color.astype(int)
        color = color.tolist()
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Emotion_Window', bgr_image)
    cv2.waitKey()
    cv2.imwrite(image_name+emotion_text+'.jpg',bgr_image)
