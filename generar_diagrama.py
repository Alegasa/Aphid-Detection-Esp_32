import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, Dense, Input
from tensorflow.keras.utils import plot_model
import os

# Agregamos esto por si el instalador no se añadió al PATH automáticamente
# Cambia la ruta si instalaste Graphviz en otra carpeta
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

# Arquitectura limpia con capa Input
model = Sequential([
    Input(shape=(96, 96, 1)),
    Conv2D(16, (3,3), strides=(2,2), activation='relu'),
    Conv2D(32, (3,3), strides=(2,2), activation='relu'),
    Conv2D(64, (3,3), strides=(2,2), activation='relu'),
    Conv2D(64, (3,3), strides=(2,2), activation='relu'),
    Conv2D(64, (3,3), strides=(1,1), activation='relu'),
    Conv2D(64, (3,3), strides=(2,2), activation='relu'),
    Flatten(),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Generar el diagrama
try:
    plot_model(
        model, 
        to_file='arquitectura_final.png', 
        show_shapes=True, 
        show_layer_names=True, 
        dpi=300
    )
    print("¡Éxito! Se ha generado 'arquitectura_final.png' en alta resolución.")
except Exception as e:
    print(f"Error al generar: {e}")