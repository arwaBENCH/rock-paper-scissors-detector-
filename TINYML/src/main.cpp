#include "esp_camera.h"*
#include "Arduino.h"

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    14
#define XCLK_GPIO_NUM     27
#define SIOD_GPIO_NUM     21
#define SIOC_GPIO_NUM     22

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       33
#define Y6_GPIO_NUM       13
#define Y5_GPIO_NUM       18
#define Y4_GPIO_NUM       5
#define Y3_GPIO_NUM       4
#define Y2_GPIO_NUM       16
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     19

void setup() {
  Serial.begin(115200);
  Serial.println("Testing OV7670 (Low RAM mode)...");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;

  config.xclk_freq_hz = 10000000;           // 10 MHz is safe for OV7670
  config.pixel_format = PIXFORMAT_RGB565;   // OV7670 does not support JPEG
  config.frame_size   = FRAMESIZE_QQVGA;    // *** very small frame ***
  config.fb_count     = 1;                  // *** only one frame buffer ***

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed! Error 0x%x\n", err);
    return;
  }
  Serial.println("Camera initialized!");

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Frame capture failed!");
    return;
  }

  Serial.printf("Captured frame: %dx%d (%d bytes)\n", fb->width, fb->height, fb->len);

  esp_camera_fb_return(fb);
}

void loop() {}
