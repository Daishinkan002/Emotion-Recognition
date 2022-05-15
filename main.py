import cv2
import numpy
import sys
import numpy as np
import approve_face
import emotion_detector


def image_captured(img_index):

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        sys.exit("Cannot Open Camera (check camera using Cheese)")

    while(True):
        #print("In Function")
        ret,frame = cap.read()

        if not ret:
            sys.exit("Can't Recieve frame window")
        
        cv2.imshow('Camera',frame)
        key = cv2.waitKey(1)
    
        if key == ord('s'):
            cv2.imwrite("cap.jpg",frame)
            img_index = 0
            break
        elif key == ord('q'):
            print("\n\n.........Video Camera is Turning Off .......\n\n")
            img_index = 2
            break
        
    cap.release()
    cv2.destroyAllWindows()
    return img_index


def show_image(image_name):
    image = cv2.imread(image_name)
    if image is None:
        sys.exit("No such image is found")
    cv2.imshow("Image",image)
    k = cv2.waitKey(0)

def take_photo():
    img_index = 1
    print("To close any Image window press(q) and to save press (s) \n")
    while(img_index):
        img_index = image_captured(img_index)
        if img_index == 2:
            print("\n\n.........Program Halted ............\n\n")
            sys.exit()
        elif img_index == 0:
            show_image("cap.jpg")
            cv2.destroyWindow("Image")
            
            mood_confirm = input("Is this Image Okay for the Mood Prediction(y or n) : ")
            if mood_confirm == 'y':
                img_index = 3
                break
            else:
                again = input("Do you want to take again (y or n) : ")
                if again == 'n':
                    sys.exit("\n\n________Thankyou for Visiting Us________\n\n")
                else:
                    img_index = 1


if __name__ == "__main__":
    
    approval = 1
    face_detection_model_path = 'models/face_detection/haarcascade_frontalface_alt.xml'
    choice = int(input("\n\n1.Take Photo\n2.Upload Photo\n\nEnter your choice : "))
    if choice == 1:
        while(approval):
            take_photo()
            approval = approve_face.count_face('cap.jpg', face_detection_model_path)
            print("No. of face Detected = ",approval)
            if(approval == 0):
                again = input("Do you want to take photo again ?(y or n) : ")
                if again == 'y':
                    approval = 1
                    continue
                else:
                    sys.exit("\n\n.............Thankyou for Visiting Us.........\n\n")
            else:
                break
        emotion_detector.predict('cap.jpg')

    elif choice == 2:
        image_name = input("Enter Image Name : ")
        img = cv2.imread(image_name)
        if(type(img) is np.ndarray):
            approval = approve_face.count_face(image_name, face_detection_model_path)
            print("No. of faces found = ",approval)
            if(approval == 0):
                print("\n\nImage is not suitable for predicting Emotions\n\n")
            else:
                emotion_detector.predict(image_name)
        else:
            print("\n\n No Image Found\n\n")
    else:
        print("\nEnter Valid Choice\n\n")
        