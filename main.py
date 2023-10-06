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

model = load_model('keras_model.h5') # carrega o modelo
capture = cv2.VideoCapture(0) # captura a webcam
hands_obj = mp.solutions.hands.Hands(max_num_hands=1) # define o número máximo de mãos
data_for_prediction = np.ndarray(    # cria um array para a imagem
    shape=(QT_IMAGES, IMG_WIDTH, IMG_HEIGHT, QT_COLOR_CHANNELS),
    dtype=np.float32
)

while True:
    _, frame = capture.read() # lê o frame
    frame = cv2.flip(frame, 1) # inverte o frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # converte o frame para RGB
    processing_results = hands_obj.process(frame_rgb) # processa o frame
    landmark_list_list = processing_results.multi_hand_landmarks

    if landmark_list_list:
        landmark_list = landmark_list_list[0]
        height, width, _ = frame.shape
        x_max = -float('inf')
        y_max = -float('inf')
        x_min = float('inf')
        y_min = float('inf')

        for landmark in landmark_list.landmark: # define os pontos máximos e mínimos
            x, y = int(landmark.x * width), int(landmark.y * height)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
        cv2.rectangle(frame, (x_min-50, y_min-50), (x_max+50, y_max+50), (0, 255, 0), 2) # desenha o retângulo

        try:
            cropped_image = frame[y_min-50:y_max+50, x_min-50:x_max+50] # recorta a imagem
            resized_image = cv2.resize(cropped_image, (224,224)) # redimensiona a imagem
            image_array = np.asarray(resized_image) # converte a imagem para array
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1 # normaliza a imagem
            data_for_prediction[0] = normalized_image_array # define a imagem para o array
            prediction = model.predict(data_for_prediction) # faz a predição
            predicted_class_idx = np.argmax(prediction) # define a classe com maior probabilidade
            cv2.putText(frame, SIGN_CLASSES[predicted_class_idx], (x_min-50,y_min-65), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), 5) # escreve a classe
        except:
            continue

    cv2.imshow('Imagem', frame) # mostra o frame
    cv2.waitKey(1) # espera uma tecla ser pressionada
