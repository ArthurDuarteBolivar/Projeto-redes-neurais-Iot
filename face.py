import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

# Suprimir mensagens de aviso
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Diretório do conjunto de dados
dataset_dir = 'dataset'

# Tamanho do lote e forma das imagens
BATCH_SIZE = 32
IMG_SHAPE  = 150  

# Gerador de imagens
image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Dividir dados em treinamento e validação
train_data_gen = image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=dataset_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='categorical',
    subset='training',
    seed=42  # adicione seed para garantir a reprodutibilidade
)

val_data_gen = image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=dataset_dir,
    shuffle=False,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='categorical',
    subset='validation',
    seed=42
)

# Criar e compilar o modelo
num_classes = len(train_data_gen.class_indices)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Treinar o modelo
EPOCHS = 100
history = model.fit(
    train_data_gen,
    epochs=EPOCHS,
    validation_data=val_data_gen
)


# Salvar o modelo treinado
model.save('./face', save_format='tf')

