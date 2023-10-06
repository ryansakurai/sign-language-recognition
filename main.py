import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import string

SIGN_CLASSES = tuple(string.ascii_uppercase)
QT_IMAGES = 1
IMG_WIDTH = 224
IMG_HEIGHT = 224
QT_COLOR_CHANNELS = 3

model = load_model('keras_model.h5')
capture = cv2.VideoCapture(0)   # 0 -> stardard device for capture
hands_obj = mp.solutions.hands.Hands(max_num_hands=1)
data_for_prediction = np.ndarray(
    shape=(QT_IMAGES, IMG_WIDTH, IMG_HEIGHT, QT_COLOR_CHANNELS),
    dtype=np.float32
)

while True:
    _, frame = capture.read()
    frame = cv2.flip(frame, 1)  # flips horizontally
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processing_results = hands_obj.process(frame_rgb)
    landmark_list_list = processing_results.multi_hand_landmarks

    if landmark_list_list:
        landmark_list = landmark_list_list[0]   # because the program only recognizes 1 hand
        height, width, _ = frame.shape
        x_max = -float('inf')
        y_max = -float('inf')
        x_min = float('inf')
        y_min = float('inf')

        for landmark in landmark_list.landmark:
            x, y = int(landmark.x * width), int(landmark.y * height)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
        cv2.rectangle(
            img=frame,
            pt1=(x_min-50, y_min-50),
            pt2=(x_max+50, y_max+50),
            color=(0, 255, 0),
            thickness=2
        )

        try:
            cropped_image = frame[y_min-50:y_max+50, x_min-50:x_max+50]
            resized_image = cv2.resize(cropped_image, (224,224))
            image_array = np.asarray(resized_image)
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            data_for_prediction[0] = normalized_image_array
            prediction = model.predict(data_for_prediction)
            predicted_class_idx = np.argmax(prediction)
            cv2.putText(
                img=frame,
                text=SIGN_CLASSES[predicted_class_idx],
                org=(x_min-50,y_min-65),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=3,
                color=(0,0,255),
                thickness=5
            )
        except:
            continue

    cv2.imshow('Image', frame)
    cv2.waitKey(1)  # to slow down frame output to normal rate
