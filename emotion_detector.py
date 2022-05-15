from statistics import mode
import cv2
import sys
from keras.models import load_model
import numpy as np
from dataset import get_labels, load_detection_model
from Image_Updation import get_face_coordinates, draw_bounding_box, draw_text, resize_image_to_fit_screen
from data_preprocessing import preprocess_input








detection_model_path = 'models/face_detection/haarcascade_frontalface_alt.xml'
emotion_model_path = 'models/emotion_detection/cnn'
emotion_labels = get_labels()

frame_window = 10

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
face_cascade = cv2.CascadeClassifier()

if not face_cascade.load(cv2.samples.findFile(detection_model_path)):
    print("Error loading face cascade file")
    exit(0)
emotion_target_size = emotion_classifier.input_shape[1:3]
print('Emotion target size = ', emotion_target_size)

emotion_window = []

def predict(image_name):
    img = cv2.imread(image_name)
    orig_shape_x, orig_shape_y, _ = img.shape
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
    for face_coordinates in faces:
        x1, x2, y1, y2 = get_face_coordinates(face_coordinates)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            sys.exit("\nFace is too small for predicting\n\n")
            
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
        # print(emotion_mode)
        draw_text(face_coordinates, rgb_image, emotion_text,
                  color, 0, -45, 1, 1)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    screen_fit_image = resize_image_to_fit_screen(bgr_image, orig_shape_x, orig_shape_y)
    cv2.imshow('Emotions', screen_fit_image)
    cv2.waitKey(0)
    cv2.imwrite('Test_Images/' + image_name+emotion_text+'.jpg',bgr_image)


# image_name = 'Test_Images/multiple_happy.jpeg'
# image_name = 'Test_Images/sad2.jpg'
# image_name = 'Test_Images/unknown.jpeg'
# image_name = 'Test_Images/disgust.jpeg'
image_name = 'Test_Images/surprise.jpg'
predict(image_name)