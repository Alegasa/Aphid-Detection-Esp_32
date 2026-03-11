import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import random

# --- 0. EL CANDADO ABSOLUTO ---
# Propósito: Garantizar la reproducibilidad estricta.
# Al fijar las semillas, aseguramos que la precisión dependa del diseño 
# de la red (arquitectura) y no de variaciones aleatorias en cada ejecución.
SEMILLA = 77
os.environ['PYTHONHASHSEED'] = str(SEMILLA)
random.seed(SEMILLA)
np.random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)

# --- 1. CONFIGURACIÓN Y CARGA (100% DATASET) ---
RUTA_DATASET = os.path.join(os.getcwd(), 'DS-LimonModificado') 
IMG_HEIGHT = 96
IMG_WIDTH = 96
BATCH_SIZE = 32

print("CARGANDO DATASET Y APLICANDO DATA AUGMENTATION...")

# Cargamos el 100% de los datos para el entrenamiento final (sin validation_split).
# Esto es ideal cuando la arquitectura ya está pulida y queremos que 
# el modelo final absorba toda la información posible antes de ir a producción.
train_ds = tf.keras.utils.image_dataset_from_directory(
    RUTA_DATASET,
    seed=SEMILLA, image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, color_mode='grayscale', 
    crop_to_aspect_ratio=True, verbose=1)

class_names = train_ds.class_names
print(f"Clases detectadas: {class_names}")

# --- MAGIA 1: DATA AUGMENTATION Y NORMALIZACIÓN ---
# Propósito: Evitar el sobreajuste (overfitting) rotando/invirtiendo imágenes,
# obligando al modelo a aprender la plaga y no memorizar las fotos.
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", seed=SEMILLA),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 2. ARQUITECTURA "ALL-CONV" ---
model = models.Sequential([
    layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    
    # Reducciones espaciales progresivas: 96 -> 48 -> 24 -> 12 -> 6 -> 3
    layers.Conv2D(32, 3, padding='same', activation='relu', strides=2),
    layers.Conv2D(64, 3, padding='same', activation='relu', strides=2),
    layers.Conv2D(128, 3, padding='same', activation='relu', strides=2),
    layers.Conv2D(128, 3, padding='same', activation='relu', strides=2),
    layers.Conv2D(128, 3, padding='same', activation='relu', strides=2),
    layers.Conv2D(128, 3, padding='valid', activation='relu', strides=1),

    layers.Flatten(),
    layers.Dropout(0.4), 
    layers.Dense(len(class_names))
])

optimizador = tf.keras.optimizers.Adam(learning_rate=0.0005)

model.compile(optimizer=optimizador, 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# --- 3. ENTRENAMIENTO ---
print("\nENTRENANDO ARQUITECTURA CON EL 100% DE LOS DATOS...")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)
historial = model.fit(train_ds, epochs=150, callbacks=[early_stopping], verbose=1)

# --- 4. EVALUACIÓN FINAL DEL MODELO ORIGINAL FLOTANTE ---
print("\nEVALUANDO PRECISIÓN DEL MODELO ORIGINAL (Flotante de 32 bits)...")
loss, accuracy = model.evaluate(train_ds, verbose=0)

print("-" * 50)
print(f"Accuracy Original Flotante (Training Data): {accuracy:.4f} ({(accuracy*100):.2f}%)")
print("-" * 50)