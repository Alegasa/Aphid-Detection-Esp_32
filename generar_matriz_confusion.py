import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
# Cambia este nombre si quieres evaluar otro archivo (ej. resultados_modelo_2.csv)
ARCHIVO_CSV = 'ESP32_CSV/resultados_modelo_1.csv'

print(f"🔄 Leyendo el archivo {ARCHIVO_CSV}...")

# ==========================================
# 2. LEER LOS DATOS DEL CSV
# ==========================================
# on_bad_lines='skip' evita que el programa se caiga si alguna fila en el Excel tiene una coma de más
df = pd.read_csv(ARCHIVO_CSV, on_bad_lines='skip')

# Extraer las columnas que nos importan y quitar filas vacías (por si las hay)
y_true = df['Clase Real'].dropna()
y_pred = df['Predicción'].dropna()

# Detectar automáticamente los nombres de tus clases (ej. 'aphid', 'healthy')
class_names = sorted(list(set(y_true) | set(y_pred)))

# Calcular la precisión global para ponerla en el título del gráfico
acierto = (y_true == y_pred).mean() * 100
print(f"✅ Precisión calculada desde el CSV: {acierto:.2f}%")

# ==========================================
# 3. GENERAR Y MOSTRAR LA MATRIZ
# ==========================================
print("📊 Dibujando la matriz de confusión...")
cm = confusion_matrix(y_true, y_pred, labels=class_names)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)

plt.title(f'Matriz de Confusión\n{ARCHIVO_CSV} (Precisión: {acierto:.2f}%)')
plt.xlabel('Predicción del Modelo')
plt.ylabel('Clase Real')
plt.tight_layout()

# ==========================================
# 4. GUARDAR LA IMAGEN
# ==========================================
# Le quitamos el .csv al nombre para guardar la imagen bonita
nombre_limpio = os.path.basename(ARCHIVO_CSV).replace(".csv", "")
nombre_imagen = f'matriz_{nombre_limpio}.png'

plt.savefig(nombre_imagen)
plt.show()

print(f"💾 ¡Listo! La imagen se guardó correctamente como '{nombre_imagen}'")