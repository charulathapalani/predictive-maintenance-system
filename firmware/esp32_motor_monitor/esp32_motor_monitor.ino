/*
 * ESP32 Data Acquisition / Edge Device
 * ------------------------------------
 * Reads condition-monitoring sensors, timestamps each reading, and sends a
 * window of 64 readings as JSON to the ingest API (api/ingest_server.py).
 *
 * IMPORTANT: pin assignments and CALIBRATION constants below are placeholders.
 * Calibrate every sensor against a reference instrument before relying on values.
 *
 * Suggested sensors (all on ADC1 so Wi-Fi does not disturb readings):
 *   GPIO34  vibration   - analog accelerometer (e.g. ADXL335 / piezo module)
 *   GPIO35  temperature - 10k NTC thermistor + 10k divider
 *   GPIO32  current     - ACS712 / SCT-013 conditioning circuit
 *   GPIO33  voltage     - ZMPT101B voltage module
 *   GPIO36  pressure    - analog pressure transducer (optional)
 *   GPIO27  speed       - Hall / IR tacho pulse input
 *
 * SAFETY: mains-voltage sensing must use isolated modules and be wired by a
 * qualified person.
 */
#include <WiFi.h>
#include <HTTPClient.h>

// ---------- User settings ----------
const char* WIFI_SSID   = "YOUR_WIFI_SSID";
const char* WIFI_PASS   = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL  = "http://192.168.1.10:5000/ingest";   // PC running the API
const char* DEVICE_ID   = "esp32-motor-01";

constexpr int      WINDOW_SIZE   = 64;     // must match src/config.py
constexpr uint32_t PERIOD_MS     = 200;    // must match SAMPLE_PERIOD_S (0.2 s)
constexpr int      PULSES_PER_REV = 1;

// ---------- Pins ----------
constexpr int PIN_VIB = 34, PIN_TEMP = 35, PIN_CUR = 32, PIN_VOLT = 33, PIN_PRES = 36, PIN_SPEED = 27;

// ---------- Calibration (PLACEHOLDERS - calibrate!) ----------
constexpr float VIB_MMS_PER_V   = 10.0;    // mm/s per volt (AC RMS)
constexpr float CUR_A_PER_V     = 10.0;    // amps per volt (AC RMS), e.g. ACS712-20A = 1/0.1
constexpr float VOLT_V_PER_V    = 400.0;   // mains volts per sensor volt (AC RMS)
constexpr float PRES_BAR_PER_V  = 3.0;     // bar per volt
constexpr float NTC_BETA = 3950.0, NTC_R0 = 10000.0, NTC_T0 = 298.15;

volatile uint32_t pulseCount = 0;
void IRAM_ATTR onPulse() { pulseCount++; }

struct Reading { uint32_t t; float vib, temp, cur, volt, speed, pres; };
Reading buf[WINDOW_SIZE];

float adcVolts(int pin) { return analogRead(pin) / 4095.0f * 3.3f; }

// RMS of the AC component over ~1 mains cycle (vibration, current, voltage)
float acRms(int pin, int n = 200, int gapUs = 100) {
  static float tmp[200];
  float mean = 0;
  for (int i = 0; i < n; i++) { tmp[i] = adcVolts(pin); mean += tmp[i]; delayMicroseconds(gapUs); }
  mean /= n;
  float sq = 0;
  for (int i = 0; i < n; i++) sq += (tmp[i] - mean) * (tmp[i] - mean);
  return sqrtf(sq / n);
}

float readTempC() {
  float v = adcVolts(PIN_TEMP);
  if (v < 0.05f || v > 3.25f) return NAN;
  float r = NTC_R0 * v / (3.3f - v);                       // thermistor to GND
  return 1.0f / (1.0f / NTC_T0 + logf(r / NTC_R0) / NTC_BETA) - 273.15f;
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED) { delay(400); Serial.print("."); }
  Serial.printf("\nIP: %s\n", WiFi.localIP().toString().c_str());
}

void sendWindow() {
  String body;
  body.reserve(WINDOW_SIZE * 150);
  body = String("{\"device_id\":\"") + DEVICE_ID + "\",\"samples\":[";
  for (int i = 0; i < WINDOW_SIZE; i++) {
    const Reading& r = buf[i];
    if (i) body += ",";
    body += "{\"timestamp\":" + String(r.t) +
            ",\"vibration\":" + String(r.vib, 3) + ",\"temperature\":" + (isnan(r.temp) ? "null" : String(r.temp, 2)) +
            ",\"current\":" + String(r.cur, 3) + ",\"voltage\":" + String(r.volt, 1) +
            ",\"speed\":" + String(r.speed, 0) + ",\"pressure\":" + String(r.pres, 3) + "}";
  }
  body += "]}";

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);
  Serial.printf("POST -> %d\n", code);
  if (code > 0) Serial.println(http.getString());   // health assessment from the server
  http.end();
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  pinMode(PIN_SPEED, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_SPEED), onPulse, RISING);
  connectWifi();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWifi();

  for (int i = 0; i < WINDOW_SIZE; i++) {
    uint32_t t0 = millis();
    pulseCount = 0;
    Reading& r = buf[i];
    r.t    = t0;
    r.vib  = acRms(PIN_VIB)  * VIB_MMS_PER_V;
    r.cur  = acRms(PIN_CUR)  * CUR_A_PER_V;
    r.volt = acRms(PIN_VOLT) * VOLT_V_PER_V;
    r.temp = readTempC();
    r.pres = adcVolts(PIN_PRES) * PRES_BAR_PER_V;
    while (millis() - t0 < PERIOD_MS) delay(1);
    r.speed = (pulseCount / (float)PULSES_PER_REV) * 60000.0f / (millis() - t0);
  }
  sendWindow();
}
