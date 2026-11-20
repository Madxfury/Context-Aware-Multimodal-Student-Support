/*
  LY_Pulse_Acquisition v2: Samvya USB wearable firmware

  Board: Seeed XIAO ESP32-S3, Arduino esp32 core 3.3.x
  Build options: USB Mode = "Hardware CDC and JTAG", USB CDC On Boot = Enabled.
  These keep the USB ID 303a:1001 that Samvya auto-detects.

  Wiring (unchanged from v1):
    I2C SDA = GPIO5 (D4), SCL = GPIO6 (D5)
      MAX30102 pulse oximeter   0x57
      MPU6050 accelerometer/gyro 0x68
      MAX30205 skin temperature 0x48-0x4F, or any address showing its power-on values
        (optional, re-detected every 5 s)
    GSR analog output = GPIO1 (A0/D0)

  Output: one line every 100 ms at 115200 baud. Fields v1 already sent keep their
  format; PULSE_Q, RED, SPO2, SPO2_RATIO, SPO2_FIT, TEMP_SENSOR and MOTION are new.
  PULSE_Q is the pulse periodicity score (see ppg_processor.h); HR is only reported
  as VALID when it is at least 0.50.
  SPO2_RATIO and SPO2_FIT are diagnostics for the last beat (red/IR ratio and red/IR
  correlation); a beat counts towards SpO2 only when the fit is at least 0.80.
  Send "RAW" (with a newline) to stream every optical sample for 10 s as
  "RAW <index> <red> <ir>" lines between RAW_START and RAW_END. Samvya ignores them.
  "PPG <rate> <index> <red> <ir> ..." lines carry the raw waveform (10 samples per
  line, 10 lines/s) so Samvya can estimate pulse rate on the computer.
  Send "SCAN" for I2C wiring diagnostics (lines between SCAN_START and SCAN_END): the
  devices on the sensor bus, whether its pull-ups are present, and any device found on
  other free pins or with SDA and SCL swapped.

  TIME=1234 ms | HR=72.4 BPM | INST=71.9 | STATUS=VALID | PULSE_Q=0.91 | IR=104233 | IR_MEAN=104100 |
  IR_SD=310 | LED=0x2A | RED=98321 | SPO2=97.8 % | SPO2_RATIO=0.512 | SPO2_FIT=0.94 |
  TEMP=33.12 C | TEMP_SENSOR=OK | GSR=2425 ADC | 2104 mV | ACC=-0.04,-1.37,8.59 |
  GYRO=0.00,-0.00,0.00 | MOTION=0.02 m/s2

  Changes from v1:
  - Heart rate: SparkFun checkForBeat() truncated IR to 16 bits, so readings above
    65,535 wrapped around. Beats are now found on the full 18-bit signal and timed
    from the sensor's sample count instead of the FIFO read time.
  - The FIFO is read directly. The library's 4-sample buffer dropped samples
    whenever the loop paused.
  - LED current is adjusted while skin is in contact, once the level has been
    steady for a second without movement. v1 calibrated it once at boot, which
    picked the lowest setting if no finger was on the sensor.
  - SpO2 is estimated from the red/IR ratio (uncalibrated, not a medical value).
    This module's red and IR channels are swapped relative to the datasheet; see
    SLOT1_IS_IR.
  - Heart rate is reported only when the beat-to-beat rate and the signal's
    repetition period agree, so weak or disturbed signals show as unavailable
    instead of as a wrong number.
  - MAX30205: searched on 0x48-0x4F and, by its power-on register values, on any other
    address; re-detected after reconnection.
    A disconnected sensor no longer leaves its last value on screen.
  - GSR millivolts are averaged like the ADC value instead of being a single sample.
  - Gyroscope calibration retries until the board is still, and the bias keeps
    adapting while the board rests.
  - MOTION reports movement with gravity removed. Acceleration magnitude at rest
    is gravity (plus any accelerometer offset), so it does not indicate movement.
  - USB output never blocks the loop when the computer is not reading.
*/

#include <Wire.h>
#include <math.h>
#include <string.h>

#include "MAX30105.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include "ppg_processor.h"

// ---------------------------------------------------------------- Pins and addresses
constexpr int SDA_PIN = 5;
constexpr int SCL_PIN = 6;
constexpr int GSR_PIN = 1;

constexpr uint8_t MAX30102_ADDR = 0x57;
constexpr uint8_t MPU6050_ADDR = 0x68;
// Most MAX30205 modules tie A0-A2 to ground (0x48) or to VDD/ground (0x48-0x4F), but the
// chip has 32 possible addresses. Other addresses are accepted when the device shows the
// MAX30205 power-on values THYST = 0x4B00 (75 C) and TOS = 0x5000 (80 C).
constexpr uint8_t MAX30205_FIRST_ADDR = 0x48;
constexpr uint8_t MAX30205_LAST_ADDR = 0x4F;
constexpr uint16_t MAX30205_THYST_DEFAULT = 0x4B00;
constexpr uint16_t MAX30205_TOS_DEFAULT = 0x5000;

// Free XIAO pins that SCAN also tries, in case a sensor is wired to them. GPIO1 (GSR) is
// left out: it is an analog input and must not be driven.
const uint8_t SCAN_PINS[] = {SDA_PIN, SCL_PIN, 2, 3, 4, 7, 8, 9, 43, 44};

// ---------------------------------------------------------------- MAX30102 settings
// 400 samples/s averaged in groups of 4 = 100 samples/s for each LED.
constexpr byte PPG_SAMPLE_AVERAGE = 4;
constexpr byte PPG_LED_MODE = 2;  // red + IR
constexpr int PPG_SAMPLE_RATE = 400;
constexpr int PPG_PULSE_WIDTH = 411;  // 18-bit samples
constexpr int PPG_ADC_RANGE = 4096;
constexpr float PPG_OUTPUT_RATE = 100.0f;

constexpr uint8_t REG_FIFO_WR_PTR = 0x04;  // followed by OVF_COUNTER and FIFO_RD_PTR
constexpr uint8_t REG_FIFO_DATA = 0x07;
constexpr int FIFO_DEPTH = 32;
constexpr int FIFO_SAMPLES_PER_READ = 20;  // 120 bytes, within the 128-byte I2C buffer

// LED 1 / FIFO slot 1 is the IR LED on this module, and LED 2 / slot 2 is red. The
// MAX30102 datasheet (and the SparkFun library) name them the other way round, but
// recordings from this board show it: on skin, slot 1 reflects ~3x more light than
// slot 2 (skin reflects more infrared than red), and the per-beat red/IR ratio is
// ~0.5 (normal SpO2) only with this mapping; the datasheet order gives ~2, which is
// physiologically impossible. Set to false for a module wired as in the datasheet.
constexpr bool SLOT1_IS_IR = true;
// LED current limits and targets are in ppg::LedController (ppg_processor.h).

// ---------------------------------------------------------------- Other settings
constexpr float MOTION_LIMIT = 1.5f;      // m/s^2 of movement that invalidates the pulse
constexpr float GYRO_STILL_NOISE = 0.02f; // rad/s standard deviation while calibrating
constexpr int GSR_WINDOW = 25;            // 0.5 s at 50 Hz
constexpr int TEMP_WINDOW = 8;            // 2 s at 4 Hz
constexpr float MAX30205_MIN_C = 0.0f;    // specified range of the MAX30205
constexpr float MAX30205_MAX_C = 50.0f;

constexpr unsigned long FIFO_INTERVAL_MS = 20;
constexpr unsigned long IMU_INTERVAL_MS = 20;
constexpr unsigned long GSR_INTERVAL_MS = 20;
constexpr unsigned long TEMP_INTERVAL_MS = 250;
constexpr unsigned long PROBE_INTERVAL_MS = 5000;
constexpr unsigned long PRINT_INTERVAL_MS = 100;
constexpr unsigned long RATE_WINDOW_MS = 5000;
constexpr unsigned long NO_LOCK_MS = 12000;

// ---------------------------------------------------------------- State
MAX30105 ppgSensor;
ppg::PulseProcessor pulse;
ppg::LedController leds;
bool ppgOK = false;
uint8_t ppgFailures = 0;
uint32_t rateSamples = 0;
unsigned long rateWindowStart = 0;
bool rateWindowGap = false;

Adafruit_MPU6050 imu;
bool imuOK = false;
float accel[3] = {0, 0, 0};
float gyro[3] = {0, 0, 0};
float gyroBias[3] = {0, 0, 0};
float gravity[3] = {0, 0, 0};
bool gravityReady = false;
float motion = 0.0f;
uint16_t stillReadings = 0;

uint16_t gsrRaw[GSR_WINDOW];
uint16_t gsrMv[GSR_WINDOW];
uint32_t gsrRawSum = 0;
uint32_t gsrMvSum = 0;
int gsrIndex = 0;
int gsrCount = 0;

bool tempOK = false;
uint8_t tempAddr = 0;
uint8_t tempFailures = 0;
float tempWindow[TEMP_WINDOW];
int tempIndex = 0;
int tempCount = 0;
float temperatureC = NAN;

// USB output recovery (see printTelemetry).
constexpr unsigned long OUTPUT_RECOVERY_MS = 1000;
bool outputBlocked = false;
unsigned long outputBlockedSince = 0;

// Diagnostics: sending "RAW" streams every optical sample for 10 s.
constexpr unsigned long RAW_STREAM_MS = 10000;
bool rawActive = false;
unsigned long rawStart = 0;
uint32_t rawIndex = 0;
char command[16];
int commandLength = 0;
void scanCommand();  // "SCAN": I2C wiring diagnostics, defined after the sensor code
void markPpgDiscontinuity();  // defined with the PPG waveform stream

unsigned long lastFifo = 0;
unsigned long lastImu = 0;
unsigned long lastGsr = 0;
unsigned long lastTemp = 0;
unsigned long lastProbe = 0;
unsigned long lastPrint = 0;

// ---------------------------------------------------------------- I2C helpers
bool i2cPresent(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

bool i2cRead(uint8_t address, uint8_t reg, uint8_t *data, size_t length) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  size_t received = Wire.requestFrom(address, length);
  if (received != length) {
    while (Wire.available()) Wire.read();
    return false;
  }
  for (size_t i = 0; i < length; i++) data[i] = Wire.read();
  return true;
}

bool i2cWrite(uint8_t address, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

// ---------------------------------------------------------------- MAX30102
// The library's "Red" setter drives LED 1 and its "IR" setter drives LED 2.
void writeLeds() {
  ppgSensor.setPulseAmplitudeRed(SLOT1_IS_IR ? leds.ir() : leds.red());
  ppgSensor.setPulseAmplitudeIR(SLOT1_IS_IR ? leds.red() : leds.ir());
}

bool startPpg() {
  if (!ppgSensor.begin(Wire, I2C_SPEED_FAST, MAX30102_ADDR)) return false;
  ppgSensor.setup(ppg::kLedDefault, PPG_SAMPLE_AVERAGE, PPG_LED_MODE, PPG_SAMPLE_RATE, PPG_PULSE_WIDTH,
                  PPG_ADC_RANGE);
  ppgSensor.setPulseAmplitudeGreen(0);
  ppgSensor.setPulseAmplitudeProximity(0);
  leds = ppg::LedController();
  writeLeds();
  ppgSensor.clearFIFO();
  pulse.reset();
  markPpgDiscontinuity();
  pulse.setSampleRate(PPG_OUTPUT_RATE);
  ppgFailures = 0;
  rateSamples = 0;
  rateWindowStart = millis();
  rateWindowGap = false;
  return true;
}

// Continuous waveform for Samvya, which estimates pulse rate on the computer:
// "PPG <sample rate Hz> <index of first sample> <red> <ir> ..." with 10 samples per line.
// The index counts every sample. It jumps by 1000 when the waveform is discontinuous
// (FIFO overflow, sensor restart, LED change), and a line dropped because the USB buffer
// was full also leaves a gap, so the computer restarts its analysis window.
constexpr int PPG_BATCH = 10;
uint32_t ppgIndex = 0;
uint32_t ppgBatchFirst = 0;
int ppgBatchCount = 0;
uint32_t ppgBatchRed[PPG_BATCH];
uint32_t ppgBatchIr[PPG_BATCH];

void markPpgDiscontinuity() {
  ppgIndex += 1000;
  ppgBatchCount = 0;
}

void queuePpgSample(uint32_t red, uint32_t ir) {
  if (ppgBatchCount == 0) ppgBatchFirst = ppgIndex;
  ppgBatchRed[ppgBatchCount] = red;
  ppgBatchIr[ppgBatchCount] = ir;
  ppgBatchCount++;
  ppgIndex++;
  if (ppgBatchCount < PPG_BATCH) return;
  ppgBatchCount = 0;
  char line[200];
  int length = snprintf(line, sizeof line, "PPG %.2f %lu", pulse.sampleRate(), (unsigned long)ppgBatchFirst);
  for (int i = 0; i < PPG_BATCH; i++) {
    if (length <= 0 || length >= (int)sizeof line) return;
    length += snprintf(line + length, sizeof line - length, " %lu %lu", (unsigned long)ppgBatchRed[i],
                       (unsigned long)ppgBatchIr[i]);
  }
  if (length <= 0 || length >= (int)sizeof line - 1) return;
  line[length++] = '\n';
  if (Serial.availableForWrite() >= length) Serial.write((const uint8_t *)line, length);
}

// The index increases for every sample, so gaps from a full USB buffer are visible.
void streamRaw(uint32_t red, uint32_t ir) {
  char line[48];
  int length = snprintf(line, sizeof line, "RAW %lu %lu %lu\n", (unsigned long)rawIndex++, (unsigned long)red,
                        (unsigned long)ir);
  if (length > 0 && Serial.availableForWrite() >= length) Serial.write((const uint8_t *)line, length);
}

void handleCommands(unsigned long now) {
  if (rawActive && now - rawStart >= RAW_STREAM_MS) {
    rawActive = false;
    Serial.println("RAW_END");
  }
  while (Serial.available() > 0) {
    int c = Serial.read();
    if (c == '\n' || c == '\r') {
      command[commandLength] = '\0';
      if (strcmp(command, "RAW") == 0) {
        rawIndex = 0;
        rawStart = now;
        rawActive = true;
        Serial.println("RAW_START index red ir (100 samples/s for 10 s)");
      } else if (strcmp(command, "SCAN") == 0) {
        scanCommand();
      }
      commandLength = 0;
    } else if (commandLength < (int)sizeof command - 1) {
      command[commandLength++] = (char)c;
    } else {
      commandLength = 0;
    }
  }
}

void notePpgFailure() {
  if (++ppgFailures < 10) return;
  ppgOK = false;
  pulse.reset();
  Serial.println("MAX30102 stopped responding; retrying every 5 s.");
}

void readPpgFifo() {
  uint8_t pointers[3];
  if (!i2cRead(MAX30102_ADDR, REG_FIFO_WR_PTR, pointers, sizeof pointers)) {
    notePpgFailure();
    return;
  }
  ppgFailures = 0;
  int pending = (pointers[0] - pointers[2]) & 0x1F;
  if (pointers[1] & 0x1F) {
    // Samples were overwritten, so the next interval would be wrong.
    pending = FIFO_DEPTH;
    pulse.markGap(1.0f, false);
    markPpgDiscontinuity();
    rateWindowGap = true;
  }

  uint8_t data[FIFO_SAMPLES_PER_READ * 6];
  while (pending > 0) {
    int count = pending < FIFO_SAMPLES_PER_READ ? pending : FIFO_SAMPLES_PER_READ;
    if (!i2cRead(MAX30102_ADDR, REG_FIFO_DATA, data, count * 6)) {
      notePpgFailure();
      return;
    }
    unsigned long now = millis();
    for (int i = 0; i < count; i++) {
      const uint8_t *s = data + i * 6;  // LED mode 2: slot 1 bytes 0-2, slot 2 bytes 3-5
      uint32_t slot1 = (((uint32_t)s[0] << 16) | ((uint32_t)s[1] << 8) | s[2]) & 0x3FFFF;
      uint32_t slot2 = (((uint32_t)s[3] << 16) | ((uint32_t)s[4] << 8) | s[5]) & 0x3FFFF;
      uint32_t ir = SLOT1_IS_IR ? slot1 : slot2;
      uint32_t red = SLOT1_IS_IR ? slot2 : slot1;
      pulse.addSample(red, ir, now);
      queuePpgSample(red, ir);
      if (rawActive) streamRaw(red, ir);
    }
    rateSamples += count;
    pending -= count;
  }
}

// The sensor clock can differ from nominal by a few percent; BPM uses the measured rate.
void updateSampleRate(unsigned long now) {
  unsigned long elapsed = now - rateWindowStart;
  if (elapsed < RATE_WINDOW_MS) return;
  if (!rateWindowGap && rateSamples > 0) {
    float measured = rateSamples * 1000.0f / elapsed;
    if (measured > 80.0f && measured < 120.0f) {
      pulse.setSampleRate(0.8f * pulse.sampleRate() + 0.2f * measured);
    }
  }
  rateSamples = 0;
  rateWindowStart = now;
  rateWindowGap = false;
}

// Keep the optical signal in range while skin is in contact; reset when contact ends.
void adjustLeds(unsigned long now) {
  bool moving = imuOK && motion > 0.5f * MOTION_LIMIT;
  bool finger = pulse.fingerPresent();
  if (!leds.update(now, finger, pulse.irLevel(), pulse.redLevel(), moving, pulse.saturated())) return;
  writeLeds();
  markPpgDiscontinuity();
  if (finger) pulse.markGap(1.5f, true);  // the level jumps; restart the filters at the new level
}

// ---------------------------------------------------------------- MPU6050
void calibrateGyro() {
  Serial.println("Keep the wearable still: calibrating gyroscope...");
  for (int attempt = 1; attempt <= 5; attempt++) {
    const int count = 150;
    double sum[3] = {0, 0, 0};
    double sumSquares[3] = {0, 0, 0};
    for (int i = 0; i < count; i++) {
      sensors_event_t a, g, t;
      imu.getEvent(&a, &g, &t);
      const float rate[3] = {g.gyro.x, g.gyro.y, g.gyro.z};
      for (int k = 0; k < 3; k++) {
        sum[k] += rate[k];
        sumSquares[k] += (double)rate[k] * rate[k];
      }
      delay(5);
    }
    float noise = 0.0f;
    for (int k = 0; k < 3; k++) {
      double mean = sum[k] / count;
      double variance = sumSquares[k] / count - mean * mean;
      noise = fmaxf(noise, sqrtf((float)fmax(variance, 0.0)));
    }
    if (noise < GYRO_STILL_NOISE) {
      for (int k = 0; k < 3; k++) gyroBias[k] = (float)(sum[k] / count);
      Serial.printf("Gyroscope calibrated (noise %.4f rad/s, attempt %d).\n", noise, attempt);
      return;
    }
  }
  Serial.println("Gyroscope was moving; its bias will be learned while the wearable rests.");
}

bool startImu(bool calibrate) {
  if (!imu.begin(MPU6050_ADDR, &Wire)) return false;
  imu.setAccelerometerRange(MPU6050_RANGE_2_G);
  imu.setGyroRange(MPU6050_RANGE_250_DEG);
  imu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  gravityReady = false;
  motion = 0.0f;
  stillReadings = 0;
  if (calibrate) calibrateGyro();
  return true;
}

void updateImu() {
  sensors_event_t a, g, t;
  imu.getEvent(&a, &g, &t);
  const float raw[3] = {a.acceleration.x, a.acceleration.y, a.acceleration.z};
  const float rate[3] = {g.gyro.x, g.gyro.y, g.gyro.z};

  if (!gravityReady) {
    for (int k = 0; k < 3; k++) gravity[k] = raw[k];
    gravityReady = true;
  }
  float moving = 0.0f;
  for (int k = 0; k < 3; k++) {
    accel[k] = raw[k];
    gravity[k] += 0.02f * (raw[k] - gravity[k]);  // ~1 s time constant at 50 Hz
    float d = raw[k] - gravity[k];
    moving += d * d;
  }
  motion += 0.2f * (sqrtf(moving) - motion);

  bool still = motion < 0.15f;
  for (int k = 0; k < 3; k++) {
    gyro[k] = rate[k] - gyroBias[k];
    if (fabsf(gyro[k]) > 0.05f) still = false;
  }
  stillReadings = still ? (stillReadings < 65535 ? stillReadings + 1 : stillReadings) : 0;
  if (stillReadings > 100) {  // still for 2 s: refine the bias slowly
    for (int k = 0; k < 3; k++) gyroBias[k] += 0.002f * (rate[k] - gyroBias[k]);
  }

  if (motion > MOTION_LIMIT) pulse.markGap(0.5f, false);
}

// ---------------------------------------------------------------- GSR
void updateGsr() {
  uint16_t raw = analogRead(GSR_PIN);
  uint16_t mv = (uint16_t)analogReadMilliVolts(GSR_PIN);
  if (gsrCount == GSR_WINDOW) {
    gsrRawSum -= gsrRaw[gsrIndex];
    gsrMvSum -= gsrMv[gsrIndex];
  } else {
    gsrCount++;
  }
  gsrRaw[gsrIndex] = raw;
  gsrMv[gsrIndex] = mv;
  gsrRawSum += raw;
  gsrMvSum += mv;
  gsrIndex = (gsrIndex + 1) % GSR_WINDOW;
}

// ---------------------------------------------------------------- MAX30205
float readTemperatureAt(uint8_t address) {
  uint8_t data[2];
  if (!i2cRead(address, 0x00, data, sizeof data)) return NAN;
  int16_t raw = (int16_t)(((uint16_t)data[0] << 8) | data[1]);
  float celsius = raw / 256.0f;
  return celsius >= MAX30205_MIN_C && celsius <= MAX30205_MAX_C ? celsius : NAN;
}

bool readRegister16(uint8_t address, uint8_t reg, uint16_t &value) {
  uint8_t data[2];
  if (!i2cRead(address, reg, data, sizeof data)) return false;
  value = ((uint16_t)data[0] << 8) | data[1];
  return true;
}

// True when a device answers like a MAX30205 that has not been reprogrammed.
bool hasMax30205Defaults(uint8_t address) {
  uint16_t thyst, tos;
  return readRegister16(address, 0x02, thyst) && readRegister16(address, 0x03, tos) &&
         thyst == MAX30205_THYST_DEFAULT && tos == MAX30205_TOS_DEFAULT;
}

bool useTemperatureAt(uint8_t address) {
  if (!i2cWrite(address, 0x01, 0x00)) return false;  // continuous conversion, normal data format
  delay(60);                                          // first conversion takes up to 50 ms
  if (isnan(readTemperatureAt(address))) return false;
  tempAddr = address;
  tempOK = true;
  tempFailures = 0;
  tempCount = 0;
  tempIndex = 0;
  temperatureC = NAN;
  return true;
}

bool startTemperature() {
  for (uint8_t address = MAX30205_FIRST_ADDR; address <= MAX30205_LAST_ADDR; address++) {
    if (i2cPresent(address) && useTemperatureAt(address)) return true;
  }
  for (uint8_t address = 0x08; address <= 0x77; address++) {
    if (address >= MAX30205_FIRST_ADDR && address <= MAX30205_LAST_ADDR) continue;
    if (address == MAX30102_ADDR || address == MPU6050_ADDR) continue;
    if (i2cPresent(address) && hasMax30205Defaults(address) && useTemperatureAt(address)) return true;
  }
  return false;
}

void updateTemperature() {
  float value = readTemperatureAt(tempAddr);
  if (isnan(value)) {
    if (++tempFailures >= 4) {
      tempOK = false;
      temperatureC = NAN;
      Serial.println("MAX30205 stopped responding; retrying every 5 s.");
    }
    return;
  }
  tempFailures = 0;
  tempWindow[tempIndex] = value;
  tempIndex = (tempIndex + 1) % TEMP_WINDOW;
  if (tempCount < TEMP_WINDOW) tempCount++;
  float sum = 0.0f;
  for (int i = 0; i < tempCount; i++) sum += tempWindow[i];
  temperatureC = sum / tempCount;
}

// ---------------------------------------------------------------- Sensor detection
void probeSensors() {
  if (!ppgOK && (ppgOK = startPpg())) Serial.println("MAX30102 connected.");

  bool imuPresent = i2cPresent(MPU6050_ADDR);
  if (imuOK && !imuPresent) {
    imuOK = false;
    Serial.println("MPU6050 disconnected.");
  } else if (!imuOK && imuPresent && (imuOK = startImu(false))) {
    Serial.println("MPU6050 connected.");
  }

  if (!tempOK && startTemperature()) Serial.printf("MAX30205 connected at 0x%02X.\n", tempAddr);
}

void scanI2C() {
  Serial.print("I2C devices:");
  int found = 0;
  for (uint8_t address = 1; address < 127; address++) {
    if (!i2cPresent(address)) continue;
    Serial.printf(" 0x%02X", address);
    found++;
  }
  Serial.println(found ? "" : " none");
}

// Prints every responding address on the current Wire pins. Returns how many answered.
int listI2cDevices(const char *label) {
  Serial.printf("SCAN %s:", label);
  int found = 0;
  for (uint8_t address = 0x08; address <= 0x77; address++) {
    if (!i2cPresent(address)) continue;
    found++;
    const char *name = address == MAX30102_ADDR ? "MAX30102"
                       : address == MPU6050_ADDR ? "MPU6050"
                       : hasMax30205Defaults(address) ? "MAX30205"
                       : "unknown";
    Serial.printf(" 0x%02X=%s", address, name);
  }
  Serial.println(found ? "" : " none");
  return found;
}

bool idlesHigh(uint8_t pin, bool pullUp) {
  pinMode(pin, pullUp ? INPUT_PULLUP : INPUT);
  delayMicroseconds(200);
  return digitalRead(pin) == HIGH;
}

// "SCAN": I2C diagnostics. Telemetry pauses for a few seconds; the pulse filters treat
// the pause as a gap. Lists the sensor bus at 400 and 100 kHz, reports whether the bus
// pull-ups are present, then looks for devices on other free pins or with SDA and SCL
// swapped. A pin pair is only tried when both lines idle high.
void scanCommand() {
  Serial.println("SCAN_START");
  listI2cDevices("sensor bus SDA=GPIO5(D4) SCL=GPIO6(D5) 400kHz");
  Wire.setClock(100000);
  listI2cDevices("sensor bus SDA=GPIO5(D4) SCL=GPIO6(D5) 100kHz");
  Wire.end();
  Serial.printf("SCAN bus pull-ups (internal pull-ups off): SDA %s, SCL %s\n",
                idlesHigh(SDA_PIN, false) ? "high" : "LOW", idlesHigh(SCL_PIN, false) ? "high" : "LOW");

  int pairsTried = 0, pairsSkipped = 0, pairsWithDevices = 0;
  for (uint8_t sda : SCAN_PINS) {
    for (uint8_t scl : SCAN_PINS) {
      if (sda == scl || (sda == SDA_PIN && scl == SCL_PIN)) continue;
      bool idle = idlesHigh(sda, true) && idlesHigh(scl, true);
      pinMode(sda, INPUT);
      pinMode(scl, INPUT);
      if (!idle) {
        pairsSkipped++;
        continue;
      }
      pairsTried++;
      Wire.begin(sda, scl, 100000);
      Wire.setTimeOut(10);
      int found = 0;
      for (uint8_t address = 0x08; address <= 0x77; address++) {
        if (i2cPresent(address)) found++;
      }
      if (found) {
        pairsWithDevices++;
        char label[48];
        snprintf(label, sizeof label, "other pins SDA=GPIO%u SCL=GPIO%u", sda, scl);
        listI2cDevices(label);
      }
      Wire.end();
      pinMode(sda, INPUT);
      pinMode(scl, INPUT);
      delay(1);
    }
  }
  Serial.printf("SCAN other pin pairs: %d tried, %d skipped (a line held low), %d with devices\n",
                pairsTried, pairsSkipped, pairsWithDevices);

  // What is on each free pin: an unconnected pin follows the internal pull-up and pull-down.
  // A pin held high is also checked for a 1-Wire presence pulse (for example a DS18B20).
  for (uint8_t pin : SCAN_PINS) {
    if (pin == SDA_PIN || pin == SCL_PIN) continue;
    bool up = idlesHigh(pin, true);
    pinMode(pin, INPUT_PULLDOWN);
    delayMicroseconds(200);
    bool down = digitalRead(pin) == HIGH;
    pinMode(pin, INPUT);
    const char *state = up && !down ? "not connected" : up && down ? "held HIGH" : !up && !down ? "held LOW" : "unstable";
    char volts[24] = "";
    if (pin <= 10) snprintf(volts, sizeof volts, ", %lu mV", (unsigned long)analogReadMilliVolts(pin));
    char oneWire[32] = "";
    if (up && down) {
      pinMode(pin, OUTPUT_OPEN_DRAIN);
      digitalWrite(pin, LOW);
      delayMicroseconds(480);
      digitalWrite(pin, HIGH);
      delayMicroseconds(70);
      bool presence = digitalRead(pin) == LOW;
      delayMicroseconds(410);
      snprintf(oneWire, sizeof oneWire, ", 1-Wire device %s", presence ? "PRESENT" : "absent");
    }
    pinMode(pin, INPUT);
    Serial.printf("SCAN pin GPIO%u: %s%s%s\n", pin, state, volts, oneWire);
  }

  // The swapped-line scan can leave a sensor part-way through a byte, holding SDA low.
  // Nine clocks and a STOP release it before the bus is used again.
  pinMode(SDA_PIN, OUTPUT_OPEN_DRAIN);
  pinMode(SCL_PIN, OUTPUT_OPEN_DRAIN);
  digitalWrite(SDA_PIN, HIGH);
  for (int i = 0; i < 9; i++) {
    digitalWrite(SCL_PIN, LOW);
    delayMicroseconds(10);
    digitalWrite(SCL_PIN, HIGH);
    delayMicroseconds(10);
  }
  digitalWrite(SDA_PIN, LOW);
  delayMicroseconds(10);
  digitalWrite(SDA_PIN, HIGH);  // STOP: SDA rises while SCL is high
  delayMicroseconds(10);
  pinMode(SDA_PIN, INPUT);
  pinMode(SCL_PIN, INPUT);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  Wire.setTimeOut(50);
  if (!tempOK && startTemperature()) Serial.printf("MAX30205 connected at 0x%02X.\n", tempAddr);
  Serial.printf("SCAN temperature sensor: %s\n", tempOK ? "connected" : "not found");
  Serial.println("SCAN_END");
}

// ---------------------------------------------------------------- Output
const char *pulseStatus(unsigned long now) {
  if (!ppgOK) return "WAITING";
  if (!pulse.fingerPresent()) return "NO_FINGER";
  if (pulse.saturated() || (imuOK && motion > MOTION_LIMIT)) return "BAD_SIGNAL";
  if (pulse.blanking()) return "DETECTING";
  if (pulse.hrValid(now)) return "VALID";
  if (now - pulse.fingerSinceMs() > NO_LOCK_MS) return "BAD_SIGNAL";
  return "DETECTING";
}

void printTelemetry(unsigned long now) {
  char hr[16] = "---", inst[16] = "---", spo2[16] = "---", temp[16] = "---";
  char gsr[16] = "---", gsrVoltage[16] = "---", acc[48] = "---", rot[48] = "---", moving[16] = "---";
  char ratio[16] = "---", fit[16] = "---", quality[16] = "---";
  const char *status = pulseStatus(now);
  if (pulse.fingerPresent() && !isnan(pulse.lastRatio())) {
    snprintf(ratio, sizeof ratio, "%.3f", pulse.lastRatio());
    snprintf(fit, sizeof fit, "%.2f", pulse.lastFit());
  }
  float pulseQuality = pulse.fingerPresent() ? pulse.pulseQuality() : NAN;
  if (!isnan(pulseQuality)) snprintf(quality, sizeof quality, "%.2f", pulseQuality);

  if (strcmp(status, "VALID") == 0) {
    snprintf(hr, sizeof hr, "%.1f BPM", pulse.bpm());
    if (!isnan(pulse.instantBpm())) snprintf(inst, sizeof inst, "%.1f", pulse.instantBpm());
    float oxygen = pulse.spo2(now);
    if (!isnan(oxygen)) snprintf(spo2, sizeof spo2, "%.1f %%", oxygen);
  }
  if (tempOK && !isnan(temperatureC)) snprintf(temp, sizeof temp, "%.2f C", temperatureC);
  if (gsrCount > 0) {
    snprintf(gsr, sizeof gsr, "%.0f", (float)gsrRawSum / gsrCount);
    snprintf(gsrVoltage, sizeof gsrVoltage, "%.0f", (float)gsrMvSum / gsrCount);
  }
  if (imuOK) {
    snprintf(acc, sizeof acc, "%.2f,%.2f,%.2f", accel[0], accel[1], accel[2]);
    snprintf(rot, sizeof rot, "%.2f,%.2f,%.2f", gyro[0], gyro[1], gyro[2]);
    snprintf(moving, sizeof moving, "%.2f m/s2", motion);
  }

  char line[480];
  int length = snprintf(
      line, sizeof line,
      "TIME=%lu ms | HR=%s | INST=%s | STATUS=%s | PULSE_Q=%s | IR=%lu | IR_MEAN=%.0f | IR_SD=%.0f | LED=0x%02X | "
      "RED=%lu | SPO2=%s | SPO2_RATIO=%s | SPO2_FIT=%s | TEMP=%s | TEMP_SENSOR=%s | GSR=%s ADC | %s mV | "
      "ACC=%s | GYRO=%s | MOTION=%s\n",
      now, hr, inst, status, quality, (unsigned long)pulse.lastIr(), pulse.irLevel(), pulse.irRms(), leds.ir(),
      (unsigned long)pulse.lastRed(), spo2, ratio, fit, temp, tempOK ? "OK" : "MISSING", gsr, gsrVoltage, acc, rot,
      moving);
  if (length <= 0 || length >= (int)sizeof line) return;
  // Write whole lines only, and never wait for the computer to read.
  if (Serial.availableForWrite() < length) {
    // The USB driver only restarts a stalled transmit interrupt from write() or
    // flush(). If output has been blocked for a second, flush(): with a zero
    // timeout it discards the stale lines and restarts transmission. Without
    // this, a stall after a USB reset or reconnect silences the board for good.
    if (!outputBlocked) {
      outputBlocked = true;
      outputBlockedSince = now;
    } else if (now - outputBlockedSince >= OUTPUT_RECOVERY_MS) {
      Serial.flush();
      outputBlocked = false;
    }
    return;
  }
  outputBlocked = false;
  Serial.write((const uint8_t *)line, length);
}

// ---------------------------------------------------------------- Arduino entry points
void setup() {
  Serial.setTxBufferSize(2048);  // must precede begin(); default 256 bytes is shorter than one line
  Serial.setTxTimeoutMs(0);
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("LY wearable firmware v2 (Samvya) - XIAO ESP32-S3");

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  scanI2C();

  pinMode(GSR_PIN, INPUT);
  analogReadResolution(12);
  analogSetPinAttenuation(GSR_PIN, ADC_11db);

  // Calibrate the gyroscope before starting the pulse sensor so its FIFO does not overflow.
  imuOK = startImu(true);
  ppgOK = startPpg();
  tempOK = startTemperature();

  Serial.printf("MAX30102 pulse/SpO2: %s\n", ppgOK ? "connected" : "NOT FOUND (retrying every 5 s)");
  Serial.printf("MPU6050 motion:      %s\n", imuOK ? "connected" : "NOT FOUND (retrying every 5 s)");
  if (tempOK) {
    Serial.printf("MAX30205 temperature: connected at 0x%02X\n", tempAddr);
  } else {
    Serial.println("MAX30205 temperature: NOT FOUND (check VCC, GND, SDA=D4, SCL=D5; send SCAN)");
  }
  Serial.println("GSR: GPIO1 (A0)");
  Serial.println("Live telemetry follows. Place one finger gently on the MAX30102.");

  unsigned long now = millis();
  lastFifo = lastImu = lastGsr = lastTemp = lastProbe = lastPrint = now;
}

void loop() {
  unsigned long now = millis();
  handleCommands(now);

  if (ppgOK && now - lastFifo >= FIFO_INTERVAL_MS) {
    lastFifo = now;
    readPpgFifo();
    if (ppgOK) {
      updateSampleRate(now);
      adjustLeds(now);
    }
  }
  if (imuOK && now - lastImu >= IMU_INTERVAL_MS) {
    lastImu = now;
    updateImu();
  }
  if (now - lastGsr >= GSR_INTERVAL_MS) {
    lastGsr = now;
    updateGsr();
  }
  if (tempOK && now - lastTemp >= TEMP_INTERVAL_MS) {
    lastTemp = now;
    updateTemperature();
  }
  if (now - lastProbe >= PROBE_INTERVAL_MS) {
    lastProbe = now;
    probeSensors();
  }
  if (now - lastPrint >= PRINT_INTERVAL_MS) {
    lastPrint = now;
    printTelemetry(now);
  }
}
