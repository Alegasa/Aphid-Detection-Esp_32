/*
 * PROYECTO: Clasificación de Hojas de Limón (Pulgón vs Sano)
 * INSTITUCIÓN: Universidad Politécnica de Victoria (UPV)
 * VERSIÓN: 4.0 - Optimizado para Benchmarking Masivo
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <EloquentTinyML.h>
#include <new> 
#include "model.h" // Tu modelo de 16-64 filtros (Sweet Spot)

// --- CONFIGURACIÓN DE RED ---
const char* ssid = "IZZI-D724";
const char* password = "VZY5MDNJJZYY";
const char* server_ip = "192.168.0.82"; 
const int server_port = 5000;

// --- CONFIGURACIÓN DEL MODELO ---
#define IMAGE_WIDTH 96
#define IMAGE_HEIGHT 96
#define IMAGE_CHANNELS 1 
#define INPUT_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS)
#define NUMBER_OF_OUTPUTS 2 
#define TENSOR_ARENA_SIZE 128 * 1024 

// Instancia global del modelo
using EloquentML = Eloquent::TinyML::TfLite<INPUT_SIZE, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE>;
EloquentML *ml = nullptr;

// Buffers en PSRAM
uint8_t *raw_buffer = NULL;  
int8_t *input_buffer = NULL; 

const int MAX_RAW_BYTES = INPUT_SIZE;
const char* etiquetas[] = {"aphid", "healthy"}; 

// Objeto HTTP Global para Keep-Alive (Velocidad)
HTTPClient http;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n--- SISTEMA DE IDENTIFICACIÓN AGRÍCOLA (UPV) ---");

  // 1. Inicializar PSRAM
  if(!psramInit()){
      Serial.println("❌ ERROR: No se detectó PSRAM. Revisa la configuración de la placa.");
      while(1);
  }

  // 2. Reservar memoria para el intérprete en PSRAM
  void* arena_memory = ps_malloc(sizeof(EloquentML));
  ml = new (arena_memory) EloquentML();

  if (!ml->begin(model_data)) {
    Serial.println("❌ Error iniciando modelo TFLite.");
    while (1);
  }
  Serial.println("✅ Modelo cargado correctamente.");

  // 3. Reservar buffers de imagen en PSRAM
  raw_buffer = (uint8_t *)ps_malloc(MAX_RAW_BYTES + 100);
  input_buffer = (int8_t *)ps_malloc(INPUT_SIZE * sizeof(int8_t)); 

  // 4. Conexión WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { 
    delay(500); 
    Serial.print(".");
  }
  // TURBO HACK: Desactivar Sleep para descargas rápidas
  WiFi.setSleep(false); 
  Serial.println("\n✅ WiFi Conectado y en modo Máximo Rendimiento.");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    http.setReuse(true); // Reutilizar conexión para ser más veloz
    http.setTimeout(5000); 
    
    String url_get = String("http://") + server_ip + ":" + server_port + "/get-next-image";
    Serial.println("\n>> Solicitando siguiente hoja...");
    
    http.begin(url_get);
    int httpCode = http.GET();

    if (httpCode == 200) {
      WiFiClient *stream = http.getStreamPtr();
      int pos = 0;
      unsigned long start_download = millis();
      
      // Recepción de imagen optimizada (Sin delays)
      while (http.connected() && pos < MAX_RAW_BYTES) {
        size_t size = stream->available();
        if (size) {
           size_t leer = min(size, (size_t)(MAX_RAW_BYTES - pos));
           stream->readBytes(raw_buffer + pos, leer);
           pos += leer;
        }
      }
      Serial.printf("   📥 Descarga terminada en %d ms\n", (int)(millis() - start_download));

      if (pos >= MAX_RAW_BYTES) {
        // --- PRE-PROCESAMIENTO ---
        for (int i = 0; i < pos; i++) {
            input_buffer[i] = (int8_t)(raw_buffer[i] - 128);
        }

        // --- INFERENCIA (EL CORAZÓN DE LA IA) ---
        int8_t prediction[NUMBER_OF_OUTPUTS] = {0};
        uint64_t t_start = micros();
        ml->predict((uint8_t*)input_buffer, (uint8_t*)prediction);
        uint64_t t_end = micros();

        uint32_t latencia_ms = (uint32_t)((t_end - t_start) / 1000);

        // --- POST-PROCESAMIENTO ---
        int8_t max_val = -128; 
        int max_idx = 0;
        String datos_crudos = "";

        for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
            datos_crudos += String(etiquetas[i]) + ": " + String(prediction[i]) + " ";
            if(prediction[i] > max_val) { 
              max_val = prediction[i]; 
              max_idx = i; 
            }
        }
        
        Serial.printf("   🧠 PREDICCIÓN: %s | Tiempo: %d ms\n", etiquetas[max_idx], latencia_ms);
        
        // --- REPORTE AL SERVIDOR FLASK ---
        String url_post = String("http://") + server_ip + ":" + server_port + "/report-result";
        http.begin(url_post);
        http.addHeader("Content-Type", "text/plain");

        // Formato exacto para que el Python saque estadísticas
        String mensaje_final = "Resultado: " + String(etiquetas[max_idx]) + 
                               " " + datos_crudos + 
                               " Tiempo: " + String(latencia_ms);
        
        int httpResponseCode = http.POST(mensaje_final);
        
        if (httpResponseCode > 0) {
            Serial.println("   ✅ Reporte enviado.");
        }
      }
      http.end(); 
    } 
    else if (httpCode == 204) {
      Serial.println("🏁 Fin del dataset. Todas las imágenes procesadas.");
      delay(10000);
    }
  } else {
    WiFi.disconnect(); 
    WiFi.reconnect(); 
    delay(2000);
  }
}