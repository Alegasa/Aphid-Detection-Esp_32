from flask import Flask, Response, request
import os
import csv
import re
from PIL import Image, ImageOps

app = Flask(__name__)

# --- CONFIGURACIÓN ---
CARPETA_TEST = "DS-Hojas-Prueba"
IMG_HEIGHT = 96
IMG_WIDTH = 96
ARCHIVO_CSV = "resultados_promedios.csv"

# Leer todas las fotos de la carpeta (incluso dentro de subcarpetas como aphid y healthy)
lista_fotos = []
if os.path.exists(CARPETA_TEST):
    for root, dirs, files in os.walk(CARPETA_TEST):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                lista_fotos.append(os.path.join(root, f))

print(f"✅ Modo Automático: Se procesarán {len(lista_fotos)} imágenes.")

foto_actual = 0
datos_resultados = []

# Inicializar el archivo CSV con los encabezados
with open(ARCHIVO_CSV, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Nombre de Imagen", "Clase Predicha", "Confianza (%)", "Respuesta Cruda (ESP32)"])

# --- RUTAS PARA EL ESP32 ---
@app.route('/get-next-image', methods=['GET'])
def get_image():
    global foto_actual
    if foto_actual < len(lista_fotos):
        img_path = lista_fotos[foto_actual]
        nombre_archivo = os.path.basename(img_path)
        print(f"\nEnviando ({foto_actual + 1}/{len(lista_fotos)}): {nombre_archivo}")
        
        # Preprocesamiento idéntico al entrenamiento
        img = Image.open(img_path)
        img_gray = img.convert('L')
        img_optimizada = ImageOps.fit(
            img_gray, 
            (IMG_WIDTH, IMG_HEIGHT), 
            method=Image.Resampling.BILINEAR, 
            centering=(0.5, 0.5)
        )
        
        raw_bytes = img_optimizada.tobytes()
        return Response(raw_bytes, mimetype='application/octet-stream')
    else:
        return "", 204

@app.route('/report-result', methods=['POST'])
def report():
    global foto_actual
    resultado_str = request.data.decode('utf-8')
    print(f"Resultado ESP32: {resultado_str}")
    
    # 1. Extraer la clase ("Resultado: aphid")
    clase_match = re.search(r"(?:Clase|Resultado):\s*([a-zA-Z0-9_]+)", resultado_str)
    clase_pred = clase_match.group(1) if clase_match else "Desconocida"
    
    # 2. Extraer los números crudos ("aphid: 50 healthy: -50")
    aphid_match = re.search(r"aphid:\s*(-?\d+)", resultado_str)
    healthy_match = re.search(r"healthy:\s*(-?\d+)", resultado_str)
    
    confianza = 0.0
    if aphid_match and healthy_match:
        val_aphid = int(aphid_match.group(1))
        val_healthy = int(healthy_match.group(1))
        
        # Tomamos el valor de la clase que ganó
        max_val = val_aphid if clase_pred == "aphid" else val_healthy
        
        # Convertimos la salida Int8 (-128 a 127) a porcentaje (0 a 100%)
        confianza = ((max_val + 128) / 255.0) * 100.0
        confianza = round(confianza, 2)
    
    nombre_archivo = os.path.basename(lista_fotos[foto_actual])
    
    # Guardar en el CSV
    with open(ARCHIVO_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([nombre_archivo, clase_pred, confianza, resultado_str])
        
    datos_resultados.append({"clase": clase_pred, "confianza": confianza})
    
    foto_actual += 1
    
    if foto_actual == len(lista_fotos):
        calcular_promedios_finales()
        
    return "OK"

def calcular_promedios_finales():
    if not datos_resultados:
        return
        
    total_confianza = sum(r['confianza'] for r in datos_resultados)
    promedio_general = total_confianza / len(datos_resultados) if len(datos_resultados) > 0 else 0
    
    conteo_clases = {}
    for r in datos_resultados:
        conteo_clases[r['clase']] = conteo_clases.get(r['clase'], 0) + 1
        
    print("\n" + "="*50)
    print("¡PROCESAMIENTO MASIVO TERMINADO!")
    print(f"Total procesadas: {len(datos_resultados)}")
    print(f"Promedio de Confianza Global: {promedio_general:.2f}%")
    print(f"Resumen de detecciones: {conteo_clases}")
    print(f"Los resultados se han guardado en: {ARCHIVO_CSV}")
    print("="*50 + "\n")
    
    # Agregar el bloque de resumen al final del CSV
    with open(ARCHIVO_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([]) 
        writer.writerow(["--- RESUMEN FINAL ---", "", "", ""])
        writer.writerow(["Total de Imágenes", len(datos_resultados), "", ""])
        writer.writerow(["Promedio Confianza General", f"{promedio_general:.2f}%", "", ""])
        
        for clase, conteo in conteo_clases.items():
            writer.writerow([f"Total detectado como '{clase}'", conteo, "", ""])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)