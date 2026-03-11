import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score

# --- 0 ---
# Garantiza la reproducibilidad estricta.
SEMILLA = 77
os.environ['PYTHONHASHSEED'] = str(SEMILLA)
random.seed(SEMILLA)
np.random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)

# --- 1. CONFIGURACIÓN ---
RUTA_DATASET = os.path.join(os.getcwd(), 'DS-LimonModificado') 
IMG_HEIGHT = 96
IMG_WIDTH = 96
BATCH_SIZE = 32

print("CARGANDO DATASET Y APLICANDO DATA AUGMENTATION...")

# Cargamos el 100% de los datos en train_ds (sin validation_split)
train_ds = tf.keras.utils.image_dataset_from_directory(
    RUTA_DATASET,
    seed=SEMILLA, image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, color_mode='grayscale', 
    crop_to_aspect_ratio=True, verbose=1)

class_names = train_ds.class_names
print(f"Clases detectadas: {class_names}")

# --- 1: DATA AUGMENTATION ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", seed=SEMILLA),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 2. ARQUITECTURA "ALL-CONV" ---
# Elimina las capas de MaxPooling usando strides=2 en las convoluciones.
# Esto reduce la carga computacional y mantiene la información espacial, ideal para TinyML.
model = models.Sequential([
    layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    
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
print("\nENTRENANDO ARQUITECTURA ESP32 COMPATIBLE (100% DATASET)...")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)
model.fit(train_ds, epochs=150, callbacks=[early_stopping], verbose=1)

# --- 4. CUANTIZACIÓN INT8 ESTRICTA ---
print("\nCUANTIZANDO MODELO A INT8 PARA ESP32...")

# Uso de Concrete Function: Fija estáticamente el tamaño de entrada (shape).
# Fundamental para que TFLite Micro asigne memoria de forma eficiente en el ESP32.
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, IMG_HEIGHT, IMG_WIDTH, 1], model.inputs[0].dtype)
)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative Dataset: Pasa imágenes reales durante la conversión para calibrar 
# el rango dinámico de las activaciones y mapear los float32 a int8 con precisión.
def representative_data_gen():
    for input_value, _ in train_ds.unbatch().batch(1).take(150):
        yield [tf.cast(input_value, tf.float32)] 
        
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  
converter.inference_output_type = tf.int8 
tflite_model = converter.convert()

# --- 5. EVALUACIÓN FINAL DEL MODELO CUANTIZADO ---
# Se aplica el Scale y Zero Point extraídos del modelo para cuantizar 
# las imágenes antes de la inferencia. (Ahora evaluamos sobre train_ds)
print("\nEVALUANDO PRECISIÓN DEL MODELO CUANTIZADO SOBRE EL SET DE ENTRENAMIENTO...")
x_train_eval, y_train_eval = [], []
for images, labels in train_ds.unbatch():
    x_train_eval.append(images.numpy())
    y_train_eval.append(labels.numpy())
x_train_eval = np.array(x_train_eval)
y_train_eval = np.array(y_train_eval)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
input_scale, input_zero_point = input_details["quantization"]

y_pred_mq = []
for i in range(len(x_train_eval)):
    img = np.expand_dims(x_train_eval[i], axis=0)
    # Ecuación de cuantización: q = (f / scale) + zero_point
    if input_scale > 0:
        img_quantized = (img / input_scale + input_zero_point).astype(input_details["dtype"])
    else:
        img_quantized = img.astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], img_quantized)
    interpreter.invoke()
    y_pred_mq.append(np.argmax(interpreter.get_tensor(output_details["index"])[0]))

acc_mq = accuracy_score(y_train_eval, y_pred_mq)
print("-" * 50)
print(f"Accuracy Cuantizado (M.Q - Training Data): {acc_mq:.4f}")
print(f"Scale: {input_scale} | Zero Point: {input_zero_point}")
print("-" * 50)

# --- 6. GUARDAR EL MODELO FÍSICO PARA ARDUINO ---
# Convierte los bytes del modelo .tflite a un arreglo en C++.
nombre_archivo = 'model.h'
with open(nombre_archivo, 'w') as f:
    hex_array = [f"0x{val:02x}" for val in tflite_model]
    f.write('#include <pgmspace.h>\n\n')
    f.write(f"// Input Scale: {input_scale}, Input Zero Point: {input_zero_point}\n")
    f.write(f"alignas(8) const unsigned char model_data[] PROGMEM = {{{', '.join(hex_array)}}};\n")
    f.write(f"const int model_data_len = {len(tflite_model)};\n")
print(f"\n¡Modelo guardado exitosamente como: {nombre_archivo}!")