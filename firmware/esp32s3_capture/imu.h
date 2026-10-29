// Register-level driver for the MPU6050 / MPU6500 / MPU9250 family.
//
// The module sold as "MPU6050" turned out to be an MPU6500 clone (WHO_AM_I = 0x70 instead of
// 0x68), which the standard Arduino library rejects. This driver accepts the whole family and
// configures every chip identically:
//   ±2 g, ±250 °/s, 100 Hz output rate, ≈ 42 Hz digital low-pass
//   (+ ACCEL_CONFIG2 accelerometer filter, which only the 6500/9250 have)
//
// Also provides:
//   - motion-gated gyro calibration: the zero-rate offset is measured at boot and subtracted;
//     calibration is REFUSED while the sensor moves (a moving calibration once shifted a
//     whole dataset by ≈ 50 °/s)
//   - bad-data detection: all-0x00 / all-0xFF bursts (loose wire) and frozen readings, so the
//     caller can re-initialise the sensor instead of streaming garbage
//
// The same file is used by esp32s3_capture and esp32s3_inference (Arduino needs a copy in each
// sketch folder; keep them identical).
#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <math.h>

// GPIO 8/9 (ESP32-S3 plan) are flash pins on the classic ESP32, so pins depend on the chip.
#if CONFIG_IDF_TARGET_ESP32S3
#define IMU_SDA_PIN 8
#define IMU_SCL_PIN 9
#else
#define IMU_SDA_PIN 21
#define IMU_SCL_PIN 22
#endif

namespace imu {

const uint8_t ADDR = 0x68;  // AD0 low
const uint8_t REG_SMPLRT_DIV = 0x19;
const uint8_t REG_CONFIG = 0x1A;
const uint8_t REG_GYRO_CONFIG = 0x1B;
const uint8_t REG_ACCEL_CONFIG = 0x1C;
const uint8_t REG_ACCEL_CONFIG2 = 0x1D;  // MPU6500 / 9250 only
const uint8_t REG_ACCEL_XOUT_H = 0x3B;
const uint8_t REG_PWR_MGMT_1 = 0x6B;
const uint8_t REG_WHO_AM_I = 0x75;

const float ACCEL_LSB_PER_G = 16384.0f;  // ±2 g
const float GYRO_LSB_PER_DPS = 131.0f;   // ±250 °/s
const uint16_t FROZEN_LIMIT = 20;        // identical raw readings in a row = frozen sensor

enum Chip { CHIP_NONE, CHIP_MPU6050, CHIP_MPU6500, CHIP_MPU9250 };
enum Status { READ_OK, READ_I2C_ERROR, READ_BAD_DATA, READ_FROZEN };

struct Sample {
  float ax, ay, az;  // g
  float gx, gy, gz;  // °/s, bias removed
  float temp_c;
};

static Chip chip = CHIP_NONE;
static uint8_t who_am_i = 0;
static float gyro_bias[3] = {0, 0, 0};
static uint8_t last_raw[14];
static uint16_t frozen_count = 0;

static inline bool writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static inline bool readRegs(uint8_t reg, uint8_t* buf, uint8_t n) {
  Wire.beginTransmission(ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint16_t)ADDR, (size_t)n, true) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

static inline uint8_t readReg(uint8_t reg) {
  uint8_t v = 0xFF;
  readRegs(reg, &v, 1);
  return v;
}

static inline const char* chipName() {
  switch (chip) {
    case CHIP_MPU6050: return "MPU6050";
    case CHIP_MPU6500: return "MPU6500";
    case CHIP_MPU9250: return "MPU9250";
    default: return "none";
  }
}

// Initialise (or re-initialise) the sensor. Gyro bias is kept across re-inits.
static inline bool begin(bool start_wire = true) {
  if (start_wire) Wire.begin(IMU_SDA_PIN, IMU_SCL_PIN, 400000);
  writeReg(REG_PWR_MGMT_1, 0x80);  // device reset
  delay(100);
  if (!writeReg(REG_PWR_MGMT_1, 0x01)) {  // wake, PLL with gyro X clock
    chip = CHIP_NONE;
    return false;
  }
  delay(10);
  who_am_i = readReg(REG_WHO_AM_I);
  switch (who_am_i) {
    case 0x68: chip = CHIP_MPU6050; break;
    case 0x70: chip = CHIP_MPU6500; break;
    case 0x71:
    case 0x73: chip = CHIP_MPU9250; break;
    default: chip = CHIP_NONE; return false;
  }
  bool ok = true;
  ok &= writeReg(REG_SMPLRT_DIV, 9);   // 1 kHz / (1 + 9) = 100 Hz
  ok &= writeReg(REG_CONFIG, 0x03);    // DLPF ≈ 42 Hz gyro
  ok &= writeReg(REG_GYRO_CONFIG, 0x00);   // ±250 °/s
  ok &= writeReg(REG_ACCEL_CONFIG, 0x00);  // ±2 g
  if (chip != CHIP_MPU6050) ok &= writeReg(REG_ACCEL_CONFIG2, 0x03);  // accel DLPF ≈ 44 Hz
  ok &= readReg(REG_CONFIG) == 0x03 && readReg(REG_SMPLRT_DIV) == 9;
  frozen_count = 0;
  return ok;
}

static inline int16_t be16(const uint8_t* p) { return (int16_t)((p[0] << 8) | p[1]); }

// Read one sample (accel, temp, gyro in one 14-byte burst). Gyro bias is subtracted.
static inline Status read(Sample& s, bool apply_bias = true) {
  uint8_t b[14];
  if (!readRegs(REG_ACCEL_XOUT_H, b, 14)) return READ_I2C_ERROR;
  bool all00 = true, allFF = true, same = true;
  for (int i = 0; i < 14; i++) {
    all00 &= b[i] == 0x00;
    allFF &= b[i] == 0xFF;
    same &= b[i] == last_raw[i];
    last_raw[i] = b[i];
  }
  if (all00 || allFF) return READ_BAD_DATA;
  frozen_count = same ? frozen_count + 1 : 0;
  if (frozen_count >= FROZEN_LIMIT) return READ_FROZEN;

  s.ax = be16(b + 0) / ACCEL_LSB_PER_G;
  s.ay = be16(b + 2) / ACCEL_LSB_PER_G;
  s.az = be16(b + 4) / ACCEL_LSB_PER_G;
  int16_t t = be16(b + 6);
  s.temp_c = (chip == CHIP_MPU6050) ? t / 340.0f + 36.53f : t / 333.87f + 21.0f;
  s.gx = be16(b + 8) / GYRO_LSB_PER_DPS;
  s.gy = be16(b + 10) / GYRO_LSB_PER_DPS;
  s.gz = be16(b + 12) / GYRO_LSB_PER_DPS;
  if (apply_bias) {
    s.gx -= gyro_bias[0];
    s.gy -= gyro_bias[1];
    s.gz -= gyro_bias[2];
  }
  return READ_OK;
}

// Measure the gyro zero-rate offset over `n` samples (10 ms apart). Returns false, and keeps
// the old bias, if the sensor moved during the measurement.
static inline bool calibrateGyro(uint16_t n = 200, float max_gyro_std = 1.0f, float max_accel_std = 0.02f,
                                 float* out_std = nullptr) {
  double gs[3] = {0, 0, 0}, gss[3] = {0, 0, 0}, as[3] = {0, 0, 0}, ass[3] = {0, 0, 0};
  uint16_t got = 0;
  for (uint16_t i = 0; i < n * 2 && got < n; i++) {
    Sample s;
    if (read(s, false) == READ_OK) {
      float g[3] = {s.gx, s.gy, s.gz}, a[3] = {s.ax, s.ay, s.az};
      for (int k = 0; k < 3; k++) {
        gs[k] += g[k]; gss[k] += g[k] * g[k];
        as[k] += a[k]; ass[k] += a[k] * a[k];
      }
      got++;
    }
    delay(10);
  }
  if (got < n / 2) return false;
  float gstd = 0, astd = 0, mean[3];
  for (int k = 0; k < 3; k++) {
    mean[k] = gs[k] / got;
    gstd = fmaxf(gstd, sqrtf(fmaxf(0, gss[k] / got - mean[k] * mean[k])));
    float am = as[k] / got;
    astd = fmaxf(astd, sqrtf(fmaxf(0, ass[k] / got - am * am)));
  }
  if (out_std) *out_std = gstd;
  if (gstd > max_gyro_std || astd > max_accel_std) return false;  // moving: refuse
  for (int k = 0; k < 3; k++) gyro_bias[k] = mean[k];
  return true;
}

}  // namespace imu
