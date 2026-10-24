#include "esp_camera.h"
#include <WiFi.h>

// ========================================
// CAMERA MODEL
// ========================================

#define CAMERA_MODEL_AI_THINKER

#include "camera_pins.h"

// ========================================
// WIFI CONFIGURATION
// ========================================

// Connect to your personal hotspot
const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";

// Camera server function implemented in app_httpd.cpp
void startCameraServer();

// ========================================
// SETUP
// ========================================

void setup() {

  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  Serial.println("==============================");
  Serial.println("ESP32-CAM Starting...");
  Serial.println("==============================");

  // ========================================
  // CAMERA CONFIGURATION
  // ========================================

  camera_config_t config = {};

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;

  // Pin field names for your existing ESP32 camera library
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // ========================================
  // FRAME SIZE AND QUALITY
  // ========================================

  if (psramFound()) {

    Serial.println("PSRAM detected!");

    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;

  } else {

    Serial.println("PSRAM not detected!");

    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 15;
    config.fb_count = 1;

  }

  // ========================================
  // INITIALIZE CAMERA
  // ========================================

  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK) {

    Serial.printf(
      "Camera initialization failed: 0x%x\n",
      err
    );

    return;
  }

  Serial.println("Camera initialized successfully!");

  // ========================================
  // CAMERA SENSOR SETTINGS
  // ========================================

  sensor_t* s = esp_camera_sensor_get();

  if (s->id.PID == OV3660_PID) {

    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);

  }

  // Start with QVGA for stable streaming.
  // Increase to VGA later if needed.
  s->set_framesize(s, FRAMESIZE_QVGA);

  // ========================================
  // WIFI CONNECTION
  // ========================================

  Serial.println();
  Serial.println("==============================");
  Serial.println("Connecting to WiFi...");
  Serial.println("==============================");

  // IMPORTANT:
  // WIFI_STA connects to your hotspot.
  // WIFI_AP creates a separate ESP32 network.

  WiFi.mode(WIFI_STA);

  WiFi.setSleep(false);

  WiFi.begin(ssid, password);

  int attempts = 0;

  while (WiFi.status() != WL_CONNECTED && attempts < 40) {

    delay(500);
    Serial.print(".");

    attempts++;
  }

  Serial.println();

  // ========================================
  // CHECK WIFI STATUS
  // ========================================

  if (WiFi.status() != WL_CONNECTED) {

    Serial.println();
    Serial.println("ERROR: WiFi connection failed!");

    Serial.println("Please check:");
    Serial.println("1. Hotspot name");
    Serial.println("2. Hotspot password");
    Serial.println("3. 2.4 GHz WiFi compatibility");

    return;
  }

  Serial.println();
  Serial.println("WiFi connected successfully!");

  Serial.println();
  Serial.println("===== NETWORK INFORMATION =====");

  Serial.print("Connected SSID: ");
  Serial.println(WiFi.SSID());

  Serial.print("WiFi Status: ");
  Serial.println(WiFi.status());

  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  Serial.print("Gateway: ");
  Serial.println(WiFi.gatewayIP());

  Serial.print("Subnet Mask: ");
  Serial.println(WiFi.subnetMask());

  Serial.print("Signal Strength: ");
  Serial.print(WiFi.RSSI());
  Serial.println(" dBm");

  Serial.println("==============================");

  // ========================================
  // START CAMERA SERVER
  // ========================================

  startCameraServer();

  Serial.println();
  Serial.println("==============================");
  Serial.println("ESP32-CAM READY!");
  Serial.println("==============================");

  Serial.println();

  Serial.print("Camera Web Interface: http://");
  Serial.println(WiFi.localIP());

  Serial.print("Live Video Stream: http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/stream");

  Serial.println();
  Serial.println("Open the Camera Web Interface");
  Serial.println("and click Start Stream.");

  Serial.println("==============================");
}

// ========================================
// MAIN LOOP
// ========================================

void loop() {

  // Check WiFi connection periodically

  if (WiFi.status() != WL_CONNECTED) {

    Serial.println("WiFi disconnected. Reconnecting...");

    WiFi.reconnect();

  }

  delay(10000);
}