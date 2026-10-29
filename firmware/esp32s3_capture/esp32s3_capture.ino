// SignSense capture firmware
//
// Reads the MPU6500/MPU6050 at a fixed 100 Hz and streams every sample to the laptop over a
// Wi-Fi WebSocket (python/capture_server.py listens on ws://<laptop-ip>:8765).
//
//   MPU6500 ──I²C 400 kHz──► ESP32 ──fixed 10 ms schedule──► JSON ──WebSocket──► laptop
//            SDA 21 / SCL 22         (gyro bias removed)
//
// One JSON object per sample:
//   {"seq":1234,"timestamp_us":12345678,"ax":0.012,"ay":-0.981,"az":0.103,
//    "gx":0.41,"gy":-0.12,"gz":0.05,"temp_c":31.2}
// `seq` increases by one per sample, so the server can detect samples lost over Wi-Fi.
// Every 5 s a stats object {"type":"stats",...} is sent and a [STATS] line printed.
//
// Board: "ESP32 Dev Module" (or ESP32S3 Dev Module), upload speed 115200.
// Library: WebSockets by Markus Sattler.
// Copy secrets.h.template to secrets.h first.
//
// Serial commands (115200 baud): c = recalibrate gyro (keep the sensor still), s = stats now

#include <WiFi.h>
#include <WebSocketsClient.h>
#include "esp_timer.h"
#include "imu.h"
#include "secrets.h"

static const uint32_t PERIOD_US = 10000;           // 100 Hz
static const uint32_t RESYNC_LATE_US = 5 * PERIOD_US;
static const uint32_t STATS_MS = 5000;
static const uint16_t BAD_READS_BEFORE_REINIT = 20;

WebSocketsClient ws;
bool ws_connected = false;
volatile bool calib_requested = false;

uint32_t seq = 0;
int64_t next_sample_us = 0;

// stats for the current 5 s period
uint32_t st_samples = 0, st_sent = 0, st_dropped = 0, st_late = 0, st_bad = 0;
uint32_t total_reinit = 0;
uint16_t bad_in_row = 0;
uint32_t last_stats_ms = 0;

void onWsEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      ws_connected = true;
      Serial.printf("[WS] connected to ws://%s:%d\n", SERVER_IP, SERVER_PORT);
      break;
    case WStype_DISCONNECTED:
      if (ws_connected) Serial.println("[WS] disconnected, retrying...");
      ws_connected = false;
      break;
    case WStype_TEXT:
      // server -> device commands
      if (length >= 1 && payload[0] == 'c') calib_requested = true;  // server asked to recalibrate
      break;
    default:
      break;
  }
}

void calibrateUntilStill() {
  for (int attempt = 1;; attempt++) {
    Serial.printf("[IMU] calibrating gyro, keep the sensor STILL (attempt %d)...\n", attempt);
    float std_dps = 0;
    if (imu::calibrateGyro(200, 1.0f, 0.02f, &std_dps)) {
      Serial.printf("[IMU] gyro bias = %.2f / %.2f / %.2f deg/s\n", imu::gyro_bias[0], imu::gyro_bias[1],
                    imu::gyro_bias[2]);
      return;
    }
    Serial.printf("[IMU] sensor moved (gyro std %.2f deg/s), calibration refused\n", std_dps);
    delay(500);
  }
}

void connectWifi() {
  if (strchr(SERVER_IP, ':')) {
    Serial.println("[ERR] SERVER_IP contains ':' - put only the IP address there, the port goes in SERVER_PORT");
  }
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // Wi-Fi power save adds 100+ ms latency spikes
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("[WiFi] connecting to %s", WIFI_SSID);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
    if (millis() - t0 > 20000) {
      Serial.println("\n[WiFi] still not connected: check SSID/password and that the network is 2.4 GHz");
      t0 = millis();
    }
  }
  Serial.printf("\n[WiFi] connected, IP %s, RSSI %d dBm\n", WiFi.localIP().toString().c_str(), WiFi.RSSI());
}

void printStats() {
  float secs = (millis() - last_stats_ms) / 1000.0f;
  if (secs <= 0) secs = 1;
  char buf[200];
  snprintf(buf, sizeof(buf),
           "{\"type\":\"stats\",\"rate_hz\":%.1f,\"sent\":%u,\"dropped\":%u,\"late\":%u,\"bad_reads\":%u,"
           "\"reinit\":%u,\"rssi\":%d,\"chip\":\"%s\"}",
           st_samples / secs, st_sent, st_dropped, st_late, st_bad, total_reinit, WiFi.RSSI(), imu::chipName());
  Serial.printf("[STATS] rate %.1f Hz  sent %u  dropped %u  late %u  bad %u  reinit %u  RSSI %d  ws %s\n",
                st_samples / secs, st_sent, st_dropped, st_late, st_bad, total_reinit, WiFi.RSSI(),
                ws_connected ? "up" : "down");
  if (ws_connected) ws.sendTXT(buf);
  st_samples = st_sent = st_dropped = st_late = st_bad = 0;
  last_stats_ms = millis();
}

void handleSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'c') calib_requested = true;
    if (c == 's') printStats();
  }
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\n=== SignSense capture firmware ===");
  while (!imu::begin()) {
    Serial.printf("[IMU] not found (WHO_AM_I = 0x%02X) on SDA %d / SCL %d, retrying...\n", imu::who_am_i,
                  IMU_SDA_PIN, IMU_SCL_PIN);
    delay(1000);
  }
  Serial.printf("[IMU] %s (WHO_AM_I 0x%02X): +-2 g, +-250 deg/s, 100 Hz, DLPF 42 Hz\n", imu::chipName(),
                imu::who_am_i);
  calibrateUntilStill();

  connectWifi();
  ws.begin(SERVER_IP, SERVER_PORT, "/");
  ws.onEvent(onWsEvent);
  ws.setReconnectInterval(2000);
  ws.enableHeartbeat(15000, 3000, 2);

  last_stats_ms = millis();
  next_sample_us = esp_timer_get_time() + PERIOD_US;
}

void loop() {
  ws.loop();
  handleSerial();
  if (calib_requested) {
    calib_requested = false;
    calibrateUntilStill();
    next_sample_us = esp_timer_get_time() + PERIOD_US;
  }

  if (WiFi.status() != WL_CONNECTED) {
    ws_connected = false;
    WiFi.reconnect();
    delay(500);
    return;
  }

  int64_t now = esp_timer_get_time();
  if (now >= next_sample_us) {
    // fixed schedule: next = previous + 10 ms, so the rate never drifts; resync if far behind
    if (now - next_sample_us > RESYNC_LATE_US) {
      next_sample_us = now;
      st_late++;
    }
    next_sample_us += PERIOD_US;

    imu::Sample s;
    imu::Status status = imu::read(s);
    if (status != imu::READ_OK) {
      st_bad++;
      if (++bad_in_row >= BAD_READS_BEFORE_REINIT) {
        Serial.printf("[IMU] bad data (%s), re-initialising sensor\n",
                      status == imu::READ_FROZEN ? "frozen" : "0x00/0xFF or I2C error");
        if (imu::begin(false)) total_reinit++;
        bad_in_row = 0;
      }
      return;
    }
    bad_in_row = 0;
    st_samples++;

    char msg[200];
    int n = snprintf(msg, sizeof(msg),
                     "{\"seq\":%u,\"timestamp_us\":%lld,\"ax\":%.4f,\"ay\":%.4f,\"az\":%.4f,"
                     "\"gx\":%.3f,\"gy\":%.3f,\"gz\":%.3f,\"temp_c\":%.1f}",
                     seq, (long long)now, s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.temp_c);
    seq++;
    if (ws_connected && ws.sendTXT(msg, n)) {
      st_sent++;
    } else {
      st_dropped++;
    }
  }

  if (millis() - last_stats_ms >= STATS_MS) printStats();
}
