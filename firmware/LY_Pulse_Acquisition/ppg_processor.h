// Pulse and SpO2 processing for the MAX30102, independent of Arduino so it can be
// tested on a computer (see firmware/tests/ppg_processor_test.cpp).
//
// Input: one red and one IR sample per call at ~100 samples/s.
// Beat times come from the sample count, not from when the FIFO was read, so I2C
// bursts do not add timing jitter.

#pragma once

#include <math.h>
#include <stdint.h>

namespace ppg {

// No-finger reflection on the LY board measured 1,000-3,100 IR counts at LED 0x1F-0x7F.
// A fingertip at the default LED current reads well above 8,000.
constexpr float kFingerOn = 8000.0f;
constexpr float kFingerOff = 6000.0f;
constexpr float kSaturated = 250000.0f;          // 18-bit full scale is 262,143
constexpr float kMinIbiMs = 300.0f;              // 200 BPM
constexpr float kMaxIbiMs = 1500.0f;             // 40 BPM
// Smallest beat accepted, as a fraction of DC. Worn on the hand, real beats measured
// 0.04-0.05% of DC, so 0.02% keeps them while ignoring sensor noise. Noise that does
// pass is rejected by the pulse quality check below.
constexpr float kMinPulseFraction = 0.0002f;
constexpr float kPeakHoldSeconds = 0.15f;        // keep the highest point within 150 ms of a peak
constexpr float kEnvelopeDecay = 0.99769f;       // per sample at 100 Hz: halves in 3 s
constexpr float kMaxPulseFraction = 0.08f;       // larger swings are movement, not pulse
constexpr uint32_t kBeatTimeoutMs = 3000;
constexpr uint8_t kIbiHistory = 8;
constexpr uint8_t kRatioHistory = 5;
constexpr uint8_t kBeatsForValid = 4;
constexpr uint8_t kRatiosForSpo2 = 3;
constexpr float kMinRatioFit = 0.8f;             // red/IR correlation needed for an SpO2 beat
// A real pulse repeats: the filtered IR signal one beat period later looks like the
// signal now, and half a beat later it does not. Quality is the autocorrelation at
// one period minus any positive autocorrelation at half a period, over the last 8 s.
// Slow waves (breathing, drift) correlate at both lags and score low, as do weak or
// irregular signals (loose contact, tremor). Heart rate is only reported when the
// quality is at least kMinPulseQuality.
constexpr int kQualityDecimation = 4;            // 100 Hz -> 25 Hz history
constexpr int kQualityLength = 200;              // 8 s at 25 Hz
constexpr int kMaxQualityLags = 32;              // +/-25% around a 40 BPM period at 25 Hz
constexpr float kMinPulseQuality = 0.5f;

inline float medianOf(const float *values, uint8_t count) {
  float sorted[16];
  if (count == 0 || count > 16) return NAN;
  for (uint8_t i = 0; i < count; i++) {
    float value = values[i];
    int8_t j = (int8_t)i - 1;
    while (j >= 0 && sorted[j] > value) {
      sorted[j + 1] = sorted[j];
      j--;
    }
    sorted[j + 1] = value;
  }
  return count % 2 ? sorted[count / 2] : 0.5f * (sorted[count / 2 - 1] + sorted[count / 2]);
}

// Maxim's empirical MAX3010x reference curve. The device is uncalibrated, so the
// result is an estimate and is not a medical measurement.
inline float spo2FromRatio(float ratio) {
  float value = -45.060f * ratio * ratio + 30.354f * ratio + 94.845f;
  return value > 100.0f ? 100.0f : value;
}

// DC removal (cut-off ~0.24 Hz) followed by a two-stage low-pass (~3.7 Hz at 100 Hz).
class BandFilter {
 public:
  void invalidate() { ready_ = false; }
  float dc() const { return dc_; }

  float step(float x) {
    if (!ready_) {
      dc_ = x;
      low1_ = low2_ = 0.0f;
      ready_ = true;
    }
    dc_ += 0.015f * (x - dc_);
    low1_ += 0.3f * ((x - dc_) - low1_);
    low2_ += 0.3f * (low1_ - low2_);
    return low2_;
  }

 private:
  float dc_ = 0.0f;
  float low1_ = 0.0f;
  float low2_ = 0.0f;
  bool ready_ = false;
};

// Least-squares slope of red AC on IR AC over one beat, with a linear trend removed
// from both. Noise on red averages out, and baseline drift does not inflate the
// ratio the way a peak-to-peak estimate does. The correlation shows whether red and
// IR moved together; beats corrupted by movement have a low correlation.
class BeatFit {
 public:
  void add(float ir, float red) {
    double t = n_;
    n_ += 1.0;
    st_ += t;
    stt_ += t * t;
    sx_ += ir;
    sy_ += red;
    sxx_ += (double)ir * ir;
    syy_ += (double)red * red;
    sxy_ += (double)ir * red;
    sxt_ += ir * t;
    syt_ += red * t;
  }

  // Returns false when there are too few samples or a channel is flat.
  bool fit(float &slope, float &correlation) const {
    if (n_ < 20.0) return false;
    double ctt = stt_ - st_ * st_ / n_;
    double cxx = sxx_ - sx_ * sx_ / n_;
    double cyy = syy_ - sy_ * sy_ / n_;
    double cxy = sxy_ - sx_ * sy_ / n_;
    double cxt = sxt_ - sx_ * st_ / n_;
    double cyt = syt_ - sy_ * st_ / n_;
    if (ctt <= 0.0) return false;
    double pxx = cxx - cxt * cxt / ctt;
    double pyy = cyy - cyt * cyt / ctt;
    double pxy = cxy - cxt * cyt / ctt;
    if (pxx <= 0.0 || pyy <= 0.0) return false;
    slope = (float)(pxy / pxx);
    correlation = (float)(pxy / sqrt(pxx * pyy));
    return true;
  }

 private:
  double n_ = 0.0, st_ = 0.0, stt_ = 0.0;
  double sx_ = 0.0, sy_ = 0.0, sxx_ = 0.0, syy_ = 0.0, sxy_ = 0.0, sxt_ = 0.0, syt_ = 0.0;
};

class PulseProcessor {
 public:
  void setSampleRate(float hz) {
    if (hz > 50.0f && hz < 200.0f) sampleRate_ = hz;
  }
  float sampleRate() const { return sampleRate_; }

  void reset() {
    float rate = sampleRate_;
    *this = PulseProcessor();
    sampleRate_ = rate;
  }

  // Ignore beats for a while after a discontinuity: FIFO overflow, LED change,
  // finger placement or movement. Filters restart 0.2 s later at the new level.
  void markGap(float seconds, bool restartFilters) {
    uint32_t until = index_ + (uint32_t)(seconds * sampleRate_);
    if (until > blankUntil_) blankUntil_ = until;
    if (restartFilters) filterRestartAt_ = index_ + (uint32_t)(0.2f * sampleRate_);
    havePeak_ = false;
    pending_ = false;
    resetExtremes();
  }

  void addSample(uint32_t red, uint32_t ir, uint32_t nowMs) {
    index_++;
    lastIr_ = ir;
    lastRed_ = red;
    irLevel_ += 0.1f * ((float)ir - irLevel_);
    redLevel_ += 0.1f * ((float)red - redLevel_);
    if (ir >= kSaturated || red >= kSaturated) saturatedUntil_ = index_ + (uint32_t)(0.5f * sampleRate_);

    bool finger = finger_ ? irLevel_ >= kFingerOff : irLevel_ > kFingerOn;
    if (finger != finger_) {
      finger_ = finger;
      fingerSinceMs_ = nowMs;
      clearPulse();
      markGap(2.0f, true);
    }
    if (!finger_) return;

    if (index_ == filterRestartAt_) {
      ir_.invalidate();
      red_.invalidate();
    }
    float irAc = ir_.step((float)ir);
    float redAc = red_.step((float)red);
    irPower_ += 0.01f * (irAc * irAc - irPower_);
    beat_.add(irAc, redAc);
    recordQualitySample(irAc);

    if (ibiCount_ && nowMs - lastBeatMs_ > kBeatTimeoutMs) clearPulse();
    detectBeat(-irAc, nowMs);  // blood volume lowers IR, so the inverted signal peaks per beat
  }

  bool fingerPresent() const { return finger_; }
  bool blanking() const { return index_ < blankUntil_; }
  bool saturated() const { return index_ < saturatedUntil_; }
  uint32_t fingerSinceMs() const { return fingerSinceMs_; }
  uint32_t lastIr() const { return lastIr_; }
  uint32_t lastRed() const { return lastRed_; }
  float irLevel() const { return irLevel_; }
  float redLevel() const { return redLevel_; }
  float irRms() const { return sqrtf(irPower_); }
  float perfusionIndex() const {
    float dc = ir_.dc();
    return finger_ && dc > 0.0f ? 100.0f * envelope_ / dc : NAN;
  }

  // Diagnostics for the most recent beat: red/IR ratio and red/IR correlation.
  float lastRatio() const { return lastRatio_; }
  float lastFit() const { return lastFit_; }

  bool hrValid(uint32_t nowMs) const {
    if (!finger_ || ibiCount_ < kBeatsForValid || nowMs - lastBeatMs_ > kBeatTimeoutMs) return false;
    float quality = pulseQuality();
    return !isnan(quality) && quality >= kMinPulseQuality;
  }

  // Periodicity score of the filtered IR signal (NaN until there are enough beats and
  // history). Updated every 40 ms; see computeQuality().
  float pulseQuality() const { return quality_; }
  float bpm() const { return bpm_; }
  float instantBpm() const { return instantBpm_; }

  float spo2(uint32_t nowMs) const {
    if (!hrValid(nowMs) || ratioCount_ < kRatiosForSpo2) return NAN;
    float value = spo2FromRatio(medianOf(ratios_, ratioCount_));
    return value >= 70.0f ? value : NAN;
  }

 private:
  void clearQualityHistory() {
    historyCount_ = historyNext_ = 0;
    decimationSum_ = 0.0f;
    decimationCount_ = 0;
    highPassPrimed_ = false;
    quality_ = NAN;
  }

  // Finds the lag, within +/-25% of the beat-to-beat period, at which the signal
  // repeats best. The two rate estimates must agree within 10%, otherwise early
  // misdetected beats could pass with a wrong rate. The score is the correlation at
  // that lag minus any positive correlation at half of it (slow waves correlate at
  // both).
  float computeQuality() const {
    if (ibiCount_ < 2) return NAN;
    const float historyRate = sampleRate_ / kQualityDecimation;
    const float period = medianOf(ibis_, ibiCount_) / 1000.0f * historyRate;
    int lo = (int)(0.75f * period);
    int hi = (int)(1.25f * period) + 1;
    if (lo < 3) lo = 3;
    if (hi - lo + 1 > kMaxQualityLags || historyCount_ < hi + 1 + (int)(2.0f * historyRate)) return NAN;
    float c[kMaxQualityLags];
    int best = -1;
    for (int lag = lo; lag <= hi; lag++) {
      c[lag - lo] = correlationAt(lag);
      if (isnan(c[lag - lo])) return NAN;
      if (best < 0 || c[lag - lo] > c[best]) best = lag - lo;
    }
    // A maximum at the edge of the search means the signal repeats at another rate.
    if (best == 0 || best == hi - lo) return 0.0f;
    const float left = c[best - 1], mid = c[best], right = c[best + 1];
    const float curve = left - 2.0f * mid + right;
    const float peakLag = lo + best + (curve < 0.0f ? 0.5f * (left - right) / curve : 0.0f);
    if (fabsf(peakLag - period) > 0.10f * period) return 0.0f;
    const float half = correlationAt((int)(peakLag / 2.0f + 0.5f));
    if (isnan(half)) return NAN;
    return mid - (half > 0.0f ? half : 0.0f);
  }

  // Keeps an 8 s, 25 Hz copy of the filtered IR signal, high-passed again at 0.6 Hz
  // (two stages) so breathing and baseline drift do not dominate the autocorrelation.
  // Blanked periods (placement, LED changes, movement) restart it so transients never
  // count as pulse.
  void recordQualitySample(float irAc) {
    if (index_ < blankUntil_) {
      clearQualityHistory();
      return;
    }
    decimationSum_ += irAc;
    if (++decimationCount_ < kQualityDecimation) return;
    float x = decimationSum_ / kQualityDecimation;
    if (!highPassPrimed_) {
      stage1In_ = x;
      stage1Out_ = stage2In_ = stage2Out_ = 0.0f;
      highPassPrimed_ = true;
    }
    // First-order high-pass: alpha = 1 / (1 + 2*pi*0.6 Hz / 25 Hz)
    const float alpha = 0.869f;
    stage1Out_ = alpha * (stage1Out_ + x - stage1In_);
    stage1In_ = x;
    stage2Out_ = alpha * (stage2Out_ + stage1Out_ - stage2In_);
    stage2In_ = stage1Out_;
    history_[historyNext_] = stage2Out_;
    historyNext_ = (historyNext_ + 1) % kQualityLength;
    if (historyCount_ < kQualityLength) historyCount_++;
    decimationSum_ = 0.0f;
    decimationCount_ = 0;
    quality_ = computeQuality();
  }

  // Pearson correlation between the history and itself shifted by `lag` samples.
  // Single precision: the ESP32-S3 has a hardware FPU for float but not double.
  float correlationAt(int lag) const {
    const int pairs = historyCount_ - lag;
    if (pairs < 10) return NAN;
    const int oldest = (historyNext_ - historyCount_ + kQualityLength) % kQualityLength;
    float sa = 0.0f, sb = 0.0f, saa = 0.0f, sbb = 0.0f, sab = 0.0f;
    for (int i = 0; i < pairs; i++) {
      float a = history_[(oldest + i) % kQualityLength];
      float b = history_[(oldest + i + lag) % kQualityLength];
      sa += a;
      sb += b;
      saa += a * a;
      sbb += b * b;
      sab += a * b;
    }
    float va = saa - sa * sa / pairs;
    float vb = sbb - sb * sb / pairs;
    if (va <= 0.0f || vb <= 0.0f) return NAN;
    return (sab - sa * sb / pairs) / sqrtf(va * vb);
  }

  void clearPulse() {
    clearQualityHistory();
    ibiCount_ = ibiNext_ = 0;
    ratioCount_ = ratioNext_ = 0;
    rejectRun_ = 0;
    bpm_ = instantBpm_ = NAN;
    lastRatio_ = lastFit_ = NAN;
    envelope_ = 0.0f;
    irPower_ = 0.0f;
    havePeak_ = false;
    pending_ = false;
    resetExtremes();
  }

  void resetExtremes() { beat_ = BeatFit(); }

  void detectBeat(float s, uint32_t nowMs) {
    const float before = previous2_;
    const float candidate = previous1_;
    previous2_ = previous1_;
    previous1_ = s;
    // Envelope = largest recent peak height, halving every 3 s. Beats must reach half
    // of it. Following the largest peaks (not the accepted ones) keeps noise peaks
    // from dragging the threshold down.
    envelope_ *= kEnvelopeDecay;

    if (index_ < blankUntil_) {
      trough_ = s;
      pending_ = false;
      return;
    }

    if (candidate > before && candidate >= s) {
      const float dc = ir_.dc();
      // Parabolic interpolation gives sub-sample peak timing.
      double curve = (double)before - 2.0 * candidate + s;
      double offset = curve < 0.0 ? 0.5 * ((double)before - s) / curve : 0.0;
      if (offset > 0.5) offset = 0.5;
      if (offset < -0.5) offset = -0.5;
      double peak = (double)(index_ - 1) + offset;

      if (pending_) {
        // Noise puts several small maxima on top of one beat; keep the highest.
        if (candidate > pendingValue_) {
          pendingValue_ = candidate;
          pendingPeak_ = peak;
          pendingHeight_ = candidate - pendingTrough_;
        }
      } else {
        const float height = candidate - trough_;
        const float threshold = 0.5f * envelope_;
        if (height > envelope_ && height <= kMaxPulseFraction * dc) envelope_ = height;
        if (dc > 0.0f && height >= kMinPulseFraction * dc && height >= threshold &&
            (!havePeak_ || peak - lastPeak_ >= refractorySamples())) {
          if (height > kMaxPulseFraction * dc) {
            markGap(1.0f, false);
            trough_ = s;
            return;
          }
          pending_ = true;
          pendingValue_ = candidate;
          pendingPeak_ = peak;
          pendingHeight_ = height;
          pendingTrough_ = trough_;
          pendingUntil_ = index_ + (uint32_t)(kPeakHoldSeconds * sampleRate_);
        }
      }
    }

    if (pending_ && index_ >= pendingUntil_) {
      if (pendingHeight_ > kMaxPulseFraction * ir_.dc()) {
        markGap(1.0f, false);
        trough_ = s;
        return;
      }
      if (havePeak_) acceptInterval((float)((pendingPeak_ - lastPeak_) * 1000.0 / sampleRate_), nowMs);
      havePeak_ = true;
      lastPeak_ = pendingPeak_;
      if (pendingHeight_ > envelope_) envelope_ = pendingHeight_;
      resetExtremes();
      pending_ = false;
      trough_ = s;
      return;
    }
    if (!pending_ && s < trough_) trough_ = s;
  }

  double refractorySamples() const {
    float ms = kMinIbiMs;
    if (ibiCount_ >= 3) {
      float median = medianOf(ibis_, ibiCount_);
      if (0.6f * median > ms) ms = 0.6f * median;
    }
    return ms * sampleRate_ / 1000.0;
  }

  void acceptInterval(float ibi, uint32_t nowMs) {
    if (ibi < kMinIbiMs || ibi > kMaxIbiMs) return;  // missed or extra beat
    if (ibiCount_ >= 3) {
      float median = medianOf(ibis_, ibiCount_);
      if (fabsf(ibi - median) > 0.25f * median) {
        // Four disagreeing beats in a row mean the rate really changed: relock.
        if (++rejectRun_ < 4) return;
        ibiCount_ = ibiNext_ = 0;
      }
    }
    rejectRun_ = 0;
    ibis_[ibiNext_] = ibi;
    ibiNext_ = (ibiNext_ + 1) % kIbiHistory;
    if (ibiCount_ < kIbiHistory) ibiCount_++;
    lastBeatMs_ = nowMs;
    instantBpm_ = 60000.0f / ibi;
    if (ibiCount_ >= kBeatsForValid) bpm_ = 60000.0f / medianOf(ibis_, ibiCount_);
    addRatio();
  }

  void addRatio() {
    float irDc = ir_.dc();
    float redDc = red_.dc();
    float slope = NAN, correlation = NAN;
    lastRatio_ = lastFit_ = NAN;
    if (!(irDc > 0.0f && redDc > 0.0f) || !beat_.fit(slope, correlation)) return;
    float ratio = slope * irDc / redDc;
    lastRatio_ = ratio;
    lastFit_ = correlation;
    if (correlation < kMinRatioFit || ratio < 0.3f || ratio > 1.2f) return;
    ratios_[ratioNext_] = ratio;
    ratioNext_ = (ratioNext_ + 1) % kRatioHistory;
    if (ratioCount_ < kRatioHistory) ratioCount_++;
  }

  float sampleRate_ = 100.0f;
  uint32_t index_ = 0;
  uint32_t blankUntil_ = 0;
  uint32_t filterRestartAt_ = 0;
  uint32_t saturatedUntil_ = 0;
  uint32_t fingerSinceMs_ = 0;
  uint32_t lastBeatMs_ = 0;
  uint32_t lastIr_ = 0;
  uint32_t lastRed_ = 0;
  bool finger_ = false;
  bool havePeak_ = false;
  float irLevel_ = 0.0f;
  float redLevel_ = 0.0f;
  float irPower_ = 0.0f;
  float previous1_ = 0.0f;
  float previous2_ = 0.0f;
  float trough_ = 0.0f;
  float envelope_ = 0.0f;
  double lastPeak_ = 0.0;
  bool pending_ = false;
  float pendingValue_ = 0.0f;
  float pendingHeight_ = 0.0f;
  float pendingTrough_ = 0.0f;
  double pendingPeak_ = 0.0;
  uint32_t pendingUntil_ = 0;
  BeatFit beat_;
  BandFilter ir_;
  BandFilter red_;
  float ibis_[kIbiHistory] = {};
  uint8_t ibiCount_ = 0;
  uint8_t ibiNext_ = 0;
  uint8_t rejectRun_ = 0;
  float ratios_[kRatioHistory] = {};
  uint8_t ratioCount_ = 0;
  uint8_t ratioNext_ = 0;
  float bpm_ = NAN;
  float instantBpm_ = NAN;
  float lastRatio_ = NAN;
  float lastFit_ = NAN;
  float history_[kQualityLength] = {};
  int historyCount_ = 0;
  int historyNext_ = 0;
  float decimationSum_ = 0.0f;
  int decimationCount_ = 0;
  float quality_ = NAN;
  bool highPassPrimed_ = false;
  float stage1In_ = 0.0f;
  float stage1Out_ = 0.0f;
  float stage2In_ = 0.0f;
  float stage2Out_ = 0.0f;
};

// LED current control for the MAX30102. Keeps the optical level near the middle of
// the 18-bit range while skin is in contact, but only changes the LEDs after the
// level has been steady for a full second without movement. Changing them while
// contact is shifting chases a moving target and delays the first reading.
constexpr uint8_t kLedDefault = 0x1F;  // 6.2 mA
constexpr uint8_t kLedMin = 0x08;
constexpr uint8_t kLedMax = 0x7F;      // 25.4 mA
constexpr float kLedTarget = 100000.0f;
constexpr float kLedLow = 50000.0f;
constexpr float kLedHigh = 200000.0f;
constexpr uint32_t kLedWindowMs = 1000;
constexpr float kLedSteadyFraction = 0.10f;

class LedController {
 public:
  uint8_t ir() const { return ir_; }
  uint8_t red() const { return red_; }

  // Call regularly (every FIFO read). Returns true when ir()/red() changed and must
  // be written to the sensor. `moving` comes from the accelerometer; `saturated`
  // means a sample reached full scale and the LEDs must come down regardless.
  bool update(uint32_t nowMs, bool finger, float irLevel, float redLevel, bool moving, bool saturated) {
    if (finger != finger_) {
      finger_ = finger;
      startWindow(nowMs);
      if (!finger && (ir_ != kLedDefault || red_ != kLedDefault)) {
        ir_ = red_ = kLedDefault;
        return true;
      }
      return false;
    }
    if (!finger_) return false;

    if (irLevel < irMin_) irMin_ = irLevel;
    if (irLevel > irMax_) irMax_ = irLevel;
    if (redLevel < redMin_) redMin_ = redLevel;
    if (redLevel > redMax_) redMax_ = redLevel;
    if (moving) moved_ = true;
    if (nowMs - windowStart_ < kLedWindowMs) return false;

    bool steady = !moved_ && irMax_ - irMin_ <= kLedSteadyFraction * irMax_ &&
                  redMax_ - redMin_ <= kLedSteadyFraction * redMax_;
    uint8_t ir = ir_, red = red_;
    if (saturated) {
      // Bring the brightest part of the window back into range.
      ir = scaled(ir_, irMax_);
      red = scaled(red_, redMax_);
    } else if (steady) {
      ir = scaled(ir_, irLevel);
      red = scaled(red_, redLevel);
    }
    startWindow(nowMs);
    if (ir == ir_ && red == red_) return false;
    ir_ = ir;
    red_ = red;
    return true;
  }

  static uint8_t scaled(uint8_t led, float level) {
    if (level >= kLedLow && level <= kLedHigh) return led;
    float target = led * kLedTarget / (level > 1.0f ? level : 1.0f);
    if (target < kLedMin) return kLedMin;
    if (target > kLedMax) return kLedMax;
    return (uint8_t)(target + 0.5f);
  }

 private:
  void startWindow(uint32_t nowMs) {
    windowStart_ = nowMs;
    irMin_ = redMin_ = INFINITY;
    irMax_ = redMax_ = 0.0f;
    moved_ = false;
  }

  uint8_t ir_ = kLedDefault;
  uint8_t red_ = kLedDefault;
  bool finger_ = false;
  bool moved_ = false;
  uint32_t windowStart_ = 0;
  float irMin_ = INFINITY;
  float irMax_ = 0.0f;
  float redMin_ = INFINITY;
  float redMax_ = 0.0f;
};

}  // namespace ppg
