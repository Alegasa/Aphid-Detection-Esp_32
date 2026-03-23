import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score

# --- 0. EL CANDADO ABSOLUTO (Cambia este número para cada prueba) ---
SEMILLA = 7
os.environ['PYTHONHASHSEED'] = str(SEMILLA)
random.seed(SEMILLA)
np.random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)

# --- 1. CONFIGURACIÓN DE CARPETAS ---
RUTA_TRAIN = os.path.join(os.getcwd(), 'DS-LimonModificado')
RUTA_VAL = os.path.join(os.getcwd(), 'DS-Hojas-Prueba') # EXAMEN FINAL

IMG_HEIGHT = 96
IMG_WIDTH = 96
BATCH_SIZE = 32

print("🔄 CARGANDO DATASETS (TRAIN Y VAL SEPARADOS)...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    RUTA_TRAIN, seed=SEMILLA, image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, color_mode='grayscale', crop_to_aspect_ratio=True, verbose=1)

val_ds = tf.keras.utils.image_dataset_from_directory(
    RUTA_VAL, seed=SEMILLA, image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, color_mode='grayscale', crop_to_aspect_ratio=True, verbose=1)

class_names = train_ds.class_names

# --- 2. DATA AUGMENTATION (¡No lo quites!) ---|
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", seed=SEMILLA),
    layers.RandomContrast(1, seed=SEMILLA),    
])
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 3. ARQUITECTURA "ALL-CONV" LIGERA ---
model = models.Sequential([
    layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.Conv2D(16, 3, padding='same', activation='relu', strides=2),
    layers.Conv2D(32, 3, padding='same', activation='relu', strides=2),
    layers.Conv2D(64, 3, padding='same', activation='relu', strides=2),
    layers.Conv2D(64, 3, padding='same', activation='relu', strides=2),
    layers.Conv2D(64, 3, padding='same', activation='relu', strides=2),
    layers.Conv2D(64, 3, padding='valid', activation='relu', strides=1),
    layers.Flatten(),
    layers.Dropout(0.5), # Aumentamos el "olvido" al 50% para que no memorice
    layers.Dense(len(class_names)) 
])

optimizador = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizador, 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# --- 4. ENTRENAMIENTO ---
print("\n🚀 ENTRENANDO ARQUITECTURA...")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
model.fit(train_ds, validation_data=val_ds, epochs=150, callbacks=[early_stopping], verbose=1)

# =========================================================================
# === AQUÍ SALEN LOS DATOS PARA TU EXCEL ===
# =========================================================================

# --- 5. EVALUACIÓN 1: MODELO ORIGINAL (FLOTANTE) ---
print("\n📊 EVALUANDO PRECISIÓN DEL MODELO ORIGINAL (Flotante de 32 bits)...")
loss, accuracy = model.evaluate(val_ds, verbose=0)
print("-" * 50)
print(f"✅ Accuracy ORIGINAL FLOTANTE (Test Data): {accuracy:.4f} ({(accuracy*100):.2f}%)")
print("-" * 50)

# =========================================================================
nombre_modelo_original = 'modelo_original_.h5'
model.save(nombre_modelo_original)
print(f"\n💾 ¡Modelo original guardado exitosamente como: {nombre_modelo_original}!")
# =========================================================================

# --- 6. CUANTIZACIÓN A INT8 ---
print("\n⚖️ CUANTIZANDO MODELO A INT8 PARA ESP32 / ESP-EYE...")
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, IMG_HEIGHT, IMG_WIDTH, 1], model.inputs[0].dtype))
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for input_value, _ in train_ds.unbatch().batch(1).take(150):
        yield [tf.cast(input_value, tf.float32)] 
        
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  
converter.inference_output_type = tf.int8 
tflite_model = converter.convert()

# --- 7. EVALUACIÓN 2: MODELO CUANTIZADO (INT8) ---
print("\n📊 EVALUANDO PRECISIÓN DEL MODELO CUANTIZADO (Int8)...")
x_val, y_true = [], []
for images, labels in val_ds.unbatch():
    x_val.append(images.numpy())
    y_true.append(labels.numpy())
x_val = np.array(x_val)
y_true = np.array(y_true)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
input_scale, input_zero_point = input_details["quantization"]

y_pred_mq = []
for i in range(len(x_val)):
    img = np.expand_dims(x_val[i], axis=0)
    if input_scale > 0:
        img_quantized = (img / input_scale + input_zero_point).astype(input_details["dtype"])
    else:
        img_quantized = img.astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], img_quantized)
    interpreter.invoke()
    y_pred_mq.append(np.argmax(interpreter.get_tensor(output_details["index"])[0]))

acc_mq = accuracy_score(y_true, y_pred_mq)
print("-" * 50)
print(f"✅ Accuracy CUANTIZADO INT8 (Test Data): {acc_mq:.4f} ({(acc_mq*100):.2f}%)")
print(f"📌 Scale: {input_scale} | Zero Point: {input_zero_point}")
print("-" * 50)

# --- 8. GUARDAR EL MODELO FÍSICO (.h) ---
nombre_archivo = 'model.h'
with open(nombre_archivo, 'w') as f:
    hex_array = [f"0x{val:02x}" for val in tflite_model]
    f.write('#include <pgmspace.h>\n\n')
    f.write(f"// Input Scale: {input_scale}, Input Zero Point: {input_zero_point}\n")
    f.write(f"alignas(8) const unsigned char model_data[] PROGMEM = {{{', '.join(hex_array)}}};\n")
    f.write(f"const int model_data_len = {len(tflite_model)};\n")
print(f"\n💾 ¡Modelo guardado exitosamente como: {nombre_archivo}!")