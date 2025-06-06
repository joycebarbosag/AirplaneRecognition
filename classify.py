import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import serial
import time

# Carrega o modelo treinado
model = load_model('model/model.h5')
class_names = ['enemy', 'friend']

print("Tentando conectar ao Arduino na porta COM4...")
arduino = serial.Serial('COM4', 9600)
print("Conexão estabelecida! Aguardando Arduino reiniciar...")
time.sleep(20)  # Espera o Arduino reiniciar
print("Arduino pronto para comunicação.")

def preprocess_frame(frame):
    img = cv2.resize(frame, (180, 180))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

threshold = 0.7
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = preprocess_frame(frame)
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]

    if confidence >= threshold:
        detected_class = class_names[class_idx]
        label = f"{detected_class} ({confidence*100:.1f}%)"

        if detected_class == 'enemy':
            print("Enemy detected! Sending fire command to Arduino...")
            arduino.write(b'f')
            time.sleep(2)
    else:
        label = ""
    if label:
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Classificador Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()