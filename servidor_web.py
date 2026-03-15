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
ARCHIVO_CSV = "resultados_detallados.csv"

# Leer fotos y determinar clase real por el nombre de la carpeta
lista_fotos = []
if os.path.exists(CARPETA_TEST):
    for root, dirs, files in os.walk(CARPETA_TEST):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                path_completo = os.path.join(root, f)
                clase_real = os.path.basename(root).lower() # Toma el nombre de la carpeta (aphid/healthy)
                lista_fotos.append({"path": path_completo, "clase_real": clase_real})

print(f"✅ Modo Estadístico: Se procesarán {len(lista_fotos)} imágenes.")

foto_actual = 0
datos_resultados = []

# Encabezados del CSV (Ahora con Clase Real y Acierto)
with open(ARCHIVO_CSV, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Nombre", "Clase Real", "Predicción", "Acierto", "Confianza (%)", "Latencia (ms)"])

@app.route('/get-next-image', methods=['GET'])
def get_image():
    global foto_actual
    if foto_actual < len(lista_fotos):
        img_path = lista_fotos[foto_actual]["path"]
        print(f"\nEnviando ({foto_actual + 1}/{len(lista_fotos)}): {os.path.basename(img_path)}")
        
        img = Image.open(img_path).convert('L')
        img_optimizada = ImageOps.fit(img, (IMG_WIDTH, IMG_HEIGHT), method=Image.Resampling.BILINEAR)
        return Response(img_optimizada.tobytes(), mimetype='application/octet-stream')
    return "", 204

@app.route('/report-result', methods=['POST'])
def report():
    global foto_actual
    res_str = request.data.decode('utf-8')
    
    # 1. Extraer Predicción, Confianza y Tiempo
    clase_match = re.search(r"(?:Clase|Resultado):\s*([a-zA-Z]+)", res_str)
    clase_pred = clase_match.group(1).lower() if clase_match else "error"
    
    aphid_val = int(re.search(r"aphid:\s*(-?\d+)", res_str).group(1)) if "aphid:" in res_str else 0
    healthy_val = int(re.search(r"healthy:\s*(-?\d+)", res_str).group(1)) if "healthy:" in res_str else 0
    max_val = aphid_val if clase_pred == "aphid" else healthy_val
    confianza = round(((max_val + 128) / 255.0) * 100.0, 2)
    
    tiempo_match = re.search(r"Tiempo:\s*(\d+)", res_str)
    latencia = int(tiempo_match.group(1)) if tiempo_match else 0

    # 2. Comparar con la Realidad
    clase_real = lista_fotos[foto_actual]["clase_real"]
    es_acierto = 1 if clase_pred == clase_real else 0
    nombre_img = os.path.basename(lista_fotos[foto_actual]["path"])

    # 3. Guardar en CSV
    with open(ARCHIVO_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([nombre_img, clase_real, clase_pred, "SÍ" if es_acierto else "NO", confianza, latencia])
        
    datos_resultados.append({
        "real": clase_real, 
        "pred": clase_pred, 
        "acierto": es_acierto, 
        "confianza": confianza,
        "latencia": latencia
    })
    
    foto_actual += 1
    if foto_actual == len(lista_fotos): calcular_promedios_finales()
    return "OK"

def calcular_promedios_finales():
    if not datos_resultados: return
    
    total = len(datos_resultados)
    # Totales Reales
    real_aphid = [r for r in datos_resultados if r['real'] == 'aphid']
    real_healthy = [r for r in datos_resultados if r['real'] == 'healthy']
    
    # Aciertos por clase
    aciertos_aphid = sum(1 for r in real_aphid if r['acierto'])
    aciertos_healthy = sum(1 for r in real_healthy if r['acierto'])
    
    # Cálculos finales
    acc_global = (sum(r['acierto'] for r in datos_resultados) / total) * 100
    acc_aphid = (aciertos_aphid / len(real_aphid) * 100) if real_aphid else 0
    acc_healthy = (aciertos_healthy / len(real_healthy) * 100) if real_healthy else 0
    prom_conf = sum(r['confianza'] for r in datos_resultados) / total
    prom_lat = sum(r['latencia'] for r in datos_resultados) / total

    output = (
        f"\n{'='*60}\n"
        f"📊 REPORTE ESTADÍSTICO FINAL (Semilla {os.getenv('SEMILLA', 'N/A')})\n"
        f"{'='*60}\n"
        f"✅ ACCURACY GLOBAL: {acc_global:.2f}%\n"
        f"⏱️ LATENCIA PROMEDIO: {prom_lat:.2f} ms\n"
        f"💡 CONFIANZA PROMEDIO: {prom_conf:.2f}%\n\n"
        f"🦟 DETECCIÓN DE PULGÓN (Aphid):\n"
        f"   - Total de imágenes: {len(real_aphid)}\n"
        f"   - Aciertos: {aciertos_aphid}\n"
        f"   - Precisión por clase: {acc_aphid:.2f}%\n\n"
        f"🍃 DETECCIÓN DE HOJA SANA (Healthy):\n"
        f"   - Total de imágenes: {len(real_healthy)}\n"
        f"   - Aciertos: {aciertos_healthy}\n"
        f"   - Precisión por clase: {acc_healthy:.2f}%\n"
        f"{'='*60}\n"
    )
    print(output)
    
    with open(ARCHIVO_CSV, mode='a', newline='') as f:
        f.write(f"\nRESUMEN,,,,,,\nAccuracy Global,{acc_global:.2f}%,,,,,\n"
                f"Acc Aphid,{acc_aphid:.2f}%,,,,,\nAcc Healthy,{acc_healthy:.2f}%,,,,,\n")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)