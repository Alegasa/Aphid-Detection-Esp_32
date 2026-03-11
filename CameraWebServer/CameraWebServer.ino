/*
 * ESP32 Client - VERSIÓN GRAYSCALE CUANTIZADA (TinyML)
 * Optimizado para modelos Int8 y comunicación vía Servidor Flask
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <EloquentTinyML.h>
#include <new> 
#include "model.h" // Asegúrate de que este sea tu modelo cuantizado

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
#define TENSOR_ARENA_SIZE 120 * 1024 // Ajustado para optimizar RAM

// Instancia de EloquentTinyML configurada para la arquitectura del ESP32
using EloquentML = Eloquent::TinyML::TfLite<INPUT_SIZE, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE>;
EloquentML *ml = nullptr;

// Buffers en PSRAM
uint8_t *raw_buffer = NULL;  
int8_t *input_buffer = NULL; 

const int MAX_RAW_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS;
const char* etiquetas[] = {"aphid", "healthy"}; 

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n--- SISTEMA DE IDENTIFICACIÓN AGRÍCOLA (UPV) ---");

  // Inicializar PSRAM (Vital para el manejo de imágenes en ESP32)
  if(!psramInit()){
      Serial.println("❌ ERROR: No se detectó PSRAM.");
      while(1);
  }

  // Reservar memoria para el intérprete en la PSRAM
  void* arena_memory = ps_malloc(sizeof(EloquentML));
  ml = new (arena_memory) EloquentML();

  if (!ml->begin(model_data)) {
    Serial.println("❌ Error iniciando modelo TFLite.");
    while (1);
  }
  Serial.println("✅ Modelo Cuantizado cargado correctamente.");

  // Reservar buffers de imagen en PSRAM
  raw_buffer = (uint8_t *)ps_malloc(MAX_RAW_BYTES + 100);
  input_buffer = (int8_t *)ps_malloc(INPUT_SIZE * sizeof(int8_t)); 

  // Conexión WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { 
    delay(500); 
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi Conectado.");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.setTimeout(5000); 
    
    String url_get = String("http://") + server_ip + ":" + server_port + "/get-next-image";
    Serial.println("\n>> Solicitando siguiente hoja...");
    
    http.begin(url_get);
    int httpCode = http.GET();

    if (httpCode == 200) {
      int total_len = http.getSize();
      WiFiClient *stream = http.getStreamPtr();
      
      int pos = 0;
      unsigned long start_time = millis();
      
      // Recepción de la imagen por streaming
      while (http.connected() && (total_len > 0 || total_len == -1)) {
        size_t size = stream->available();
        if (size) {
           if (pos + size > MAX_RAW_BYTES) size = MAX_RAW_BYTES - pos;
           stream->readBytes(raw_buffer + pos, size);
           pos += size;
           if (total_len > 0) total_len -= size;
        }
        if (millis() - start_time > 5000) break; 
        delay(1);
      }

      if (pos > 0) {
        // 1. Procesamiento: Mapeo a rango Int8 (-128 a 127)
        for (int i = 0; i < pos; i++) {
            input_buffer[i] = (int8_t)(raw_buffer[i] - 128);
        }

        int8_t prediction[NUMBER_OF_OUTPUTS] = {0};
        uint64_t t_start = micros();

        // CORRECCIÓN: Usamos (uint8_t*) para engañar al compilador y que acepte el buffer
        ml->predict((uint8_t*)input_buffer, (uint8_t*)prediction);
        
        uint64_t t_end = micros();

        // 2. Determinar clase ganadora
        int8_t max_val = -128; 
        int max_idx = 0;
        String reporte = "";

        for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
            reporte += String(etiquetas[i]) + ": " + String(prediction[i]) + " ";
            if(prediction[i] > max_val) { 
              max_val = prediction[i]; 
              max_idx = i; 
            }
        }
        
        Serial.printf("   --> PREDICCIÓN: %s | Tiempo: %d ms\n", etiquetas[max_idx], (int)((t_end-t_start)/1000));
        
        // 3. Reportar al servidor Flask
        http.end();
        String url_post = String("http://") + server_ip + ":" + server_port + "/report-result";
        http.begin(url_post);
        http.POST("Resultado: " + String(etiquetas[max_idx]) + " Datos: " + reporte);
      }
      http.end();
    } else if (httpCode == 204) {
      Serial.println("   (No hay más imágenes para procesar)");
      delay(5000);
    }
  } else {
    WiFi.disconnect(); 
    WiFi.reconnect(); 
    delay(2000);
  }
}