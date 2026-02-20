// ============================================
// ESP32 PVDF PLANT HEALTH MONITORING SYSTEM
// ============================================
// Instead of:
#include <driver/adc.h>

// Use one of these:
#include <driver/adc.h>  // This should work if paths are configured
// OR if using Arduino framework:
#include <Arduino.h>
#include <esp32-hal-adc.h>

// For ESP32 ADC functions, you can also use:
#include "esp_adc_cal.h"

// ============================================
// CONFIGURATION
// ============================================
#define DEBUG_MODE true            // Set false for production
#define PVDF_PIN 34                // GPIO34 (ADC1_CH6)
#define LED_PIN 2                  // Onboard LED
#define BUTTON_PIN 0               // Boot button for calibration

// Sampling parameters
#define SAMPLE_RATE 1000           // Hz (1000 samples/second)
#define BUFFER_SIZE 2048           // 2 seconds of data
#define FFT_SIZE 1024              // FFT window size

// Calibration constants
#define CALIBRATION_SAMPLES 1000
#define HEALTHY_THRESHOLD 0.02     // 20mV RMS for healthy plant
#define STRESS_THRESHOLD 0.05      // 50mV RMS for stress

// ============================================
// GLOBAL VARIABLES
// ============================================
Preferences preferences;
ArduinoFFT<float> FFT = ArduinoFFT<float>();

// Data buffers
volatile float rawBuffer[BUFFER_SIZE];
volatile float processedBuffer[BUFFER_SIZE];
volatile int bufferIndex = 0;
volatile bool bufferReady = false;

// Sensor calibration
float baselineVoltage = 0.0;
float sensitivity = 1.0;
float noiseFloor = 0.0;

// Plant status
enum PlantStatus { UNKNOWN, HEALTHY, PEST_STRESS, WATER_STRESS, CALIBRATING };
PlantStatus currentStatus = UNKNOWN;
float healthScore = 50.0;

// Feature extraction
float features[8] = {0};

// ============================================
// SETUP FUNCTION
// ============================================
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  
  // Initialize pins
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(PVDF_PIN, INPUT);
  
  // Configure ADC
  analogReadResolution(12);          // 12-bit resolution
  analogSetAttenuation(ADC_11db);    // 0-3.3V range
  analogSetPinAttenuation(PVDF_PIN, ADC_11db);
  
  // Initialize preferences (non-volatile storage)
  preferences.begin("plant-monitor", false);
  
  // Load calibration
  loadCalibration();
  
  // Setup sampling timer
  setupSamplingTimer();
  
  // Startup sequence
  startupSequence();
  
  Serial.println("\n=========================================");
  Serial.println("🌿 ESP32 PVDF PLANT HEALTH MONITOR");
  Serial.println("=========================================");
  Serial.println("Sensor: PVDF Piezoelectric Film (28μm)");
  Serial.println("Sampling Rate: " + String(SAMPLE_RATE) + " Hz");
  Serial.println("FFT Size: " + String(FFT_SIZE));
  Serial.println("Baseline: " + String(baselineVoltage, 4) + " V");
  Serial.println("Sensitivity: " + String(sensitivity, 2));
  Serial.println("Noise Floor: " + String(noiseFloor, 4) + " V");
  Serial.println("=========================================\n");
  
  Serial.println("📋 Commands:");
  Serial.println("  'c' - Start calibration");
  Serial.println("  'r' - Reset calibration");
  Serial.println("  's' - Show status");
  Serial.println("  'd' - Toggle debug mode");
  Serial.println("=========================================\n");
}

// ============================================
// SAMPLING TIMER INTERRUPT
// ============================================
hw_timer_t *samplingTimer = NULL;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

void IRAM_ATTR onSamplingTimer() {
  portENTER_CRITICAL_ISR(&timerMux);
  
  // Read PVDF sensor
  int rawValue = analogRead(PVDF_PIN);
  float voltage = rawValue * (3.3 / 4095.0);
  
  // Remove baseline and apply sensitivity
  float processedVoltage = (voltage - baselineVoltage) * sensitivity;
  
  // Store in buffers
  rawBuffer[bufferIndex] = voltage;
  processedBuffer[bufferIndex] = processedVoltage;
  
  bufferIndex++;
  
  // Check if buffer is full
  if (bufferIndex >= BUFFER_SIZE) {
    bufferIndex = 0;
    bufferReady = true;
  }
  
  portEXIT_CRITICAL_ISR(&timerMux);
}

void setupSamplingTimer() {
  samplingTimer = timerBegin(0, 80, true);  // 80 MHz / 80 = 1 MHz
  timerAttachInterrupt(samplingTimer, &onSamplingTimer, true);
  timerAlarmWrite(samplingTimer, 1000000 / SAMPLE_RATE, true);
  timerAlarmEnable(samplingTimer);
}

// ============================================
// CALIBRATION FUNCTIONS
// ============================================
void loadCalibration() {
  baselineVoltage = preferences.getFloat("baseline", 1.65);  // Midpoint of 3.3V
  sensitivity = preferences.getFloat("sensitivity", 1.0);
  noiseFloor = preferences.getFloat("noise", 0.001);
}

void saveCalibration() {
  preferences.putFloat("baseline", baselineVoltage);
  preferences.putFloat("sensitivity", sensitivity);
  preferences.putFloat("noise", noiseFloor);
  preferences.putUInt("calibrated", 1);
}

void calibrateSensor() {
  Serial.println("\n🔧 Starting sensor calibration...");
  Serial.println("1. Ensure plant is in resting state (no wind/touch)");
  Serial.println("2. Calibration will take 10 seconds");
  Serial.println("3. Keep environment stable\n");
  
  currentStatus = CALIBRATING;
  digitalWrite(LED_PIN, HIGH);
  
  // Collect calibration data
  float sum = 0;
  float minVal = 3.3;
  float maxVal = 0;
  
  for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
    float voltage = analogRead(PVDF_PIN) * (3.3 / 4095.0);
    sum += voltage;
    
    if (voltage < minVal) minVal = voltage;
    if (voltage > maxVal) maxVal = voltage;
    
    if (i % 100 == 0) {
      Serial.print(".");
    }
    delay(10);
  }
  
  // Calculate baseline and noise
  baselineVoltage = sum / CALIBRATION_SAMPLES;
  noiseFloor = (maxVal - minVal) / 2;
  
  // Auto-adjust sensitivity
  sensitivity = 0.1 / noiseFloor;  // Target 0.1V range
  
  // Save to persistent storage
  saveCalibration();
  
  Serial.println("\n✅ Calibration complete!");
  Serial.println("Baseline: " + String(baselineVoltage, 4) + " V");
  Serial.println("Noise Floor: " + String(noiseFloor, 4) + " V");
  Serial.println("Sensitivity: " + String(sensitivity, 2));
  
  digitalWrite(LED_PIN, LOW);
  currentStatus = UNKNOWN;
}

// ============================================
// FEATURE EXTRACTION FUNCTIONS
// ============================================
void extractFeatures(float* data, int size) {
  // 1. RMS Value
  float sumSquares = 0;
  for (int i = 0; i < size; i++) {
    sumSquares += data[i] * data[i];
  }
  features[0] = sqrt(sumSquares / size) * 1000;  // Convert to mV
  
  // 2. Peak Frequency (using FFT)
  float vReal[FFT_SIZE];
  float vImag[FFT_SIZE];
  
  // Copy data for FFT
  for (int i = 0; i < FFT_SIZE; i++) {
    vReal[i] = data[i];
    vImag[i] = 0;
  }
  
  // Perform FFT
  FFT.windowing(vReal, FFT_SIZE, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.compute(vReal, vImag, FFT_SIZE, FFT_FORWARD);
  FFT.complexToMagnitude(vReal, vImag, FFT_SIZE);
  
  // Find peak frequency (skip DC)
  float peakMag = 0;
  int peakIndex = 0;
  for (int i = 1; i < FFT_SIZE / 2; i++) {
    if (vReal[i] > peakMag) {
      peakMag = vReal[i];
      peakIndex = i;
    }
  }
  features[1] = (peakIndex * SAMPLE_RATE) / FFT_SIZE;
  
  // 3. Zero Crossing Rate
  int zeroCrossings = 0;
  for (int i = 1; i < size; i++) {
    if ((data[i-1] < 0 && data[i] >= 0) || (data[i-1] >= 0 && data[i] < 0)) {
      zeroCrossings++;
    }
  }
  features[2] = (zeroCrossings * 100.0) / size;
  
  // 4. Peak-to-Peak Amplitude
  float minVal = data[0];
  float maxVal = data[0];
  for (int i = 1; i < size; i++) {
    if (data[i] < minVal) minVal = data[i];
    if (data[i] > maxVal) maxVal = data[i];
  }
  features[3] = (maxVal - minVal) * 1000;  // mV
  
  // 5. Mean Absolute Value
  float sumAbs = 0;
  for (int i = 0; i < size; i++) {
    sumAbs += abs(data[i]);
  }
  features[4] = (sumAbs / size) * 1000;  // mV
  
  // 6. Skewness
  float mean = 0;
  for (int i = 0; i < size; i++) mean += data[i];
  mean /= size;
  
  float sum3 = 0;
  float sum2 = 0;
  for (int i = 0; i < size; i++) {
    float diff = data[i] - mean;
    sum2 += diff * diff;
    sum3 += diff * diff * diff;
  }
  float variance = sum2 / size;
  float stdDev = sqrt(variance);
  
  if (stdDev > 0.001) {
    features[5] = (sum3 / size) / (stdDev * stdDev * stdDev);
  } else {
    features[5] = 0;
  }
  
  // 7. Kurtosis
  float sum4 = 0;
  for (int i = 0; i < size; i++) {
    float diff = data[i] - mean;
    sum4 += diff * diff * diff * diff;
  }
  if (variance > 0.001) {
    features[6] = (sum4 / size) / (variance * variance) - 3;
  } else {
    features[6] = 0;
  }
  
  // 8. Signal Energy
  features[7] = sumSquares * 1000;
}

// ============================================
// PLANT HEALTH ANALYSIS
// ============================================
void analyzePlantHealth() {
  // Rule-based analysis (can be replaced with AI)
  float rms = features[0] / 1000;  // Convert back to V
  
  if (rms < 0.005) {  // Very low vibration
    currentStatus = WATER_STRESS;
    healthScore = 30.0 + random(0, 200) / 10.0;
  } else if (rms > 0.1) {  // Very high vibration
    currentStatus = PEST_STRESS;
    healthScore = 40.0 + random(0, 300) / 10.0;
  } else if (rms > 0.02 && rms < 0.08) {  // Healthy range
    currentStatus = HEALTHY;
    healthScore = 80.0 + random(0, 200) / 10.0;
  } else {
    currentStatus = UNKNOWN;
    healthScore = 50.0 + random(0, 500) / 10.0;
  }
  
  // Constrain health score
  healthScore = constrain(healthScore, 0, 100);
}

// ============================================
// DATA TRANSMISSION
// ============================================
void sendToPython() {
  // Send features as CSV
  String dataString = String(millis()) + ",";
  for (int i = 0; i < 8; i++) {
    dataString += String(features[i], 4);
    if (i < 7) dataString += ",";
  }
  
  // Add status and health score
  dataString += ",";
  switch (currentStatus) {
    case HEALTHY: dataString += "0,"; break;
    case PEST_STRESS: dataString += "1,"; break;
    case WATER_STRESS: dataString += "2,"; break;
    default: dataString += "-1,"; break;
  }
  dataString += String(healthScore, 1);
  
  Serial.println(dataString);
  
  // Blink LED on data transmission
  digitalWrite(LED_PIN, HIGH);
  delay(1);
  digitalWrite(LED_PIN, LOW);
}

// ============================================
// DISPLAY FUNCTIONS
// ============================================
void displayStatus() {
  static unsigned long lastDisplay = 0;
  if (millis() - lastDisplay < 5000) return;  // Update every 5 seconds
  
  lastDisplay = millis();
  
  String statusText;
  switch (currentStatus) {
    case HEALTHY: statusText = "✅ HEALTHY"; break;
    case PEST_STRESS: statusText = "⚠️ PEST STRESS"; break;
    case WATER_STRESS: statusText = "💧 WATER STRESS"; break;
    case CALIBRATING: statusText = "🔧 CALIBRATING"; break;
    default: statusText = "❓ UNKNOWN"; break;
  }
  
  Serial.println("\n📊 PLANT STATUS");
  Serial.println("================");
  Serial.println("Status: " + statusText);
  Serial.println("Health Score: " + String(healthScore, 1) + "%");
  Serial.println("RMS: " + String(features[0], 1) + " mV");
  Serial.println("Peak Freq: " + String(features[1], 1) + " Hz");
  Serial.println("Zero Crossing: " + String(features[2], 1) + "%");
  Serial.println("Peak-Peak: " + String(features[3], 1) + " mV");
  Serial.println("================");
}

void startupSequence() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }
}

// ============================================
// MAIN LOOP
// ============================================
void loop() {
  // Check for serial commands
  if (Serial.available() > 0) {
    char command = Serial.read();
    handleCommand(command);
  }
  
  // Check calibration button
  if (digitalRead(BUTTON_PIN) == LOW) {
    delay(50);  // Debounce
    if (digitalRead(BUTTON_PIN) == LOW) {
      calibrateSensor();
      while (digitalRead(BUTTON_PIN) == LOW);  // Wait for release
    }
  }
  
  // Process data if buffer is ready
  if (bufferReady) {
    portENTER_CRITICAL(&timerMux);
    bufferReady = false;
    
    // Copy buffer for processing
    float tempBuffer[FFT_SIZE];
    for (int i = 0; i < FFT_SIZE; i++) {
      tempBuffer[i] = processedBuffer[i];
    }
    
    portEXIT_CRITICAL(&timerMux);
    
    // Extract features
    extractFeatures(tempBuffer, FFT_SIZE);
    
    // Analyze plant health
    analyzePlantHealth();
    
    // Send to Python
    sendToPython();
    
    // Display status (if debug mode)
    if (DEBUG_MODE) {
      displayStatus();
    }
  }
  
  // Small delay to prevent watchdog
  delay(1);
}

// ============================================
// COMMAND HANDLER
// ============================================
void handleCommand(char command) {
  switch (command) {
    case 'c':
      calibrateSensor();
      break;
      
    case 'r':
      preferences.clear();
      Serial.println("✅ Calibration reset");
      break;
      
    case 's':
      displayStatus();
      break;
      
    case 'd':
      DEBUG_MODE = !DEBUG_MODE;
      Serial.println("Debug mode: " + String(DEBUG_MODE ? "ON" : "OFF"));
      break;
      
    case 'h':
      Serial.println("\n📋 Available commands:");
      Serial.println("  c - Calibrate sensor");
      Serial.println("  r - Reset calibration");
      Serial.println("  s - Show status");
      Serial.println("  d - Toggle debug");
      Serial.println("  h - Show this help");
      break;
      
    default:
      Serial.println("❓ Unknown command. Type 'h' for help.");
      break;
  }
}