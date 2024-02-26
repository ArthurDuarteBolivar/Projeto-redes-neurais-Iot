import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Carregar o modelo treinado
model = tf.keras.models.load_model('face')

# Mapear a classe prevista para a etiqueta correspondente
class_labels = ['anyone', 'arthur']  # Substitua pelos nomes das suas classes

# Inicializar a câmera (pode variar dependendo da câmera)
cap = cv2.VideoCapture(0)  # 0 indica a câmera padrão, altere se necessário

while True:
    # Capturar frame da câmera
    ret, frame = cap.read()

    # Redimensionar o frame para o tamanho desejado
    frame = cv2.resize(frame, (150, 150))

    # Pré-processar a imagem
    img_array = image.img_to_array(frame)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Fazer a previsão
    predictions = model.predict(img_array)

    # Obter a classe prevista
    predicted_class = np.argmax(predictions)

    # Obter a etiqueta da classe prevista
    predicted_label = class_labels[predicted_class]

    # Exibir a previsão no frame
    cv2.putText(frame, predicted_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Exibir o frame
    cv2.imshow('Face Recognition', frame)

    # Parar a execução ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):   
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
