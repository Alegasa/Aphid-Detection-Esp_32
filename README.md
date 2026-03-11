# Aphid-Detection-Esp_32

# Entrenamiento del Modelo: Detección de Pulgón (TinyML)
Este documento detalla el pipeline de entrenamiento diseñado para crear un modelo de clasificación de imágenes altamente eficiente, capaz de ejecutarse en microcontroladores con recursos limitados como el ESP32.

El enfoque principal es maximizar la precisión utilizando una arquitectura All-Conv y comprimir el modelo mediante Cuantización a INT8.

# 1. Preparación y Flujo de Datos
El modelo se entrena utilizando el 100% del dataset disponible (DS-LimonModificado) para asegurar que la red absorba la mayor cantidad de variabilidad posible antes de ser desplegada en producción.

Resolución de Entrada: Las imágenes se redimensionan a 96x96 píxeles en escala de grises.

Data Augmentation: Se aplican transformaciones aleatorias (RandomFlip horizontal y vertical) para evitar el sobreajuste (overfitting) y forzar al modelo a aprender patrones invariantes.

# 2. Arquitectura "All-Conv" (Sin MaxPooling)
Para optimizar el uso de memoria RAM en el ESP32, se eliminaron las tradicionales capas de MaxPooling. En su lugar, la reducción espacial se logra utilizando convoluciones con pasos largos (strides=2).

# 3. Configuración del Entrenamiento
El proceso de entrenamiento base se realiza en formato de punto flotante de 32 bits (Float32).

Optimizador: Adam, con una tasa de aprendizaje (Learning Rate) amigable de 0.0005 para evitar saltos bruscos en esta red profunda.

Función de Pérdida: SparseCategoricalCrossentropy(from_logits=True).

Early Stopping: Dado que se utiliza el 100% de los datos, el callback monitorea la métrica de accuracy de entrenamiento, con una paciencia de 10 épocas para detener el proceso una vez que el modelo converge y restaurar los mejores pesos.
