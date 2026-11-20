// Host test for firmware/LY_Pulse_Acquisition/ppg_processor.h
// Build and run: clang++ -std=c++17 -O2 -I../LY_Pulse_Acquisition ppg_processor_test.cpp -o ppg_test && ./ppg_test

#include <cmath>
#include <cstdio>
#include <random>

#include "ppg_processor.h"

static int failures = 0;

#define CHECK(condition, ...)                     \
  do {                                            \
    if (!(condition)) {                           \
      failures++;                                 \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                   \
      std::printf("\n");                          \
    }                                             \
  } while (0)

// Pulse shape over one beat (phase 0..1): systolic peak plus a smaller dicrotic wave.
static double pulseShape(double phase) {
  auto bump = [](double x, double centre, double width) {
    double d = (x - centre) / width;
    return std::exp(-0.5 * d * d);
  };
  return bump(phase, 0.18, 0.07) + 0.35 * bump(phase, 0.45, 0.06);
}

struct Signal {
  double bpm = 72.0;
  double irDc = 120000.0;    // above 65,535, where the old 16-bit detector wrapped
  double redDc = 90000.0;
  double irPerfusion = 0.012;
  double ratio = 0.5;        // red perfusion / IR perfusion
  double noise = 0.0;        // counts RMS
  double wander = 0.0;       // slow baseline drift, counts
  double variability = 0.0;  // beat-to-beat rate jitter, fraction
};

struct Simulator {
  ppg::PulseProcessor pulse;
  std::mt19937 random{42};
  double phase = 0.0;
  double beatRate = 72.0;
  uint32_t sample = 0;
  double fs = 100.0;

  uint32_t nowMs() const { return (uint32_t)(sample * 1000.0 / fs); }

  void run(const Signal &signal, double seconds, bool finger = true) {
    std::normal_distribution<double> noise(0.0, signal.noise > 0 ? signal.noise : 1e-9);
    std::uniform_real_distribution<double> jitter(-signal.variability, signal.variability);
    int samples = (int)(seconds * fs);
    for (int i = 0; i < samples; i++) {
      sample++;
      double t = sample / fs;
      phase += beatRate / 60.0 / fs;
      if (phase >= 1.0) {
        phase -= 1.0;
        beatRate = signal.bpm * (1.0 + jitter(random));
      }
      double shape = pulseShape(phase);
      double drift = signal.wander * std::sin(2 * M_PI * 0.1 * t);
      double ir, red;
      if (finger) {
        ir = signal.irDc * (1.0 - signal.irPerfusion * shape) + drift + noise(random);
        red = signal.redDc * (1.0 - signal.irPerfusion * signal.ratio * shape) + drift * 0.75 + noise(random);
      } else {
        ir = 1100 + noise(random);
        red = 900 + noise(random);
      }
      pulse.addSample((uint32_t)std::fmax(0, red), (uint32_t)std::fmax(0, ir), nowMs());
    }
  }
};

static void expectRate(const char *name, const Signal &signal, double tolerance = 2.0) {
  Simulator sim;
  sim.beatRate = signal.bpm;
  sim.run(signal, 20.0);
  uint32_t now = sim.nowMs();
  CHECK(sim.pulse.hrValid(now), "%s: no valid heart rate after 20 s", name);
  CHECK(std::fabs(sim.pulse.bpm() - signal.bpm) <= tolerance, "%s: bpm %.1f, expected %.1f", name,
        sim.pulse.bpm(), signal.bpm);
  double expectedSpo2 = ppg::spo2FromRatio((float)signal.ratio);
  float spo2 = sim.pulse.spo2(now);
  CHECK(!std::isnan(spo2), "%s: no SpO2 estimate", name);
  CHECK(std::fabs(spo2 - expectedSpo2) <= 1.5, "%s: SpO2 %.1f, expected %.1f", name, spo2, expectedSpo2);
  float quality = sim.pulse.pulseQuality();
  CHECK(quality >= 0.7f, "%s: pulse quality %.2f", name, quality);
  std::printf("ok  %-34s bpm %6.1f (true %5.1f)  SpO2 %5.1f (true %5.1f)  quality %.2f\n", name, sim.pulse.bpm(),
              signal.bpm, spo2, expectedSpo2, quality);
}

int main() {
  Signal base;
  expectRate("72 BPM, IR 120k", base);

  Signal slow = base;
  slow.bpm = 48;
  expectRate("48 BPM", slow);

  Signal fast = base;
  fast.bpm = 150;
  expectRate("150 BPM", fast);

  Signal noisy = base;
  noisy.noise = 150;           // ~10% of the pulse amplitude
  noisy.wander = 3000;
  noisy.variability = 0.05;
  expectRate("noise, drift, 5% rate variability", noisy, 3.0);

  Signal saturatedRange = base;
  saturatedRange.irDc = 230000;  // near full scale, crosses several 65,536 boundaries
  saturatedRange.redDc = 200000;
  expectRate("IR 230k", saturatedRange);

  Signal weak = base;
  weak.irPerfusion = 0.002;      // 0.2% perfusion
  weak.noise = 20;
  expectRate("weak perfusion 0.2%", weak);

  Signal lowOxygen = base;
  lowOxygen.ratio = 0.9;
  expectRate("ratio 0.9 (low SpO2)", lowOxygen);

  // No finger: never valid.
  {
    Simulator sim;
    sim.run(base, 10.0, false);
    CHECK(!sim.pulse.fingerPresent(), "no finger: finger detected");
    CHECK(!sim.pulse.hrValid(sim.nowMs()), "no finger: heart rate reported");
    std::printf("ok  no finger -> no reading\n");
  }

  // Finger removed: reading clears immediately.
  {
    Simulator sim;
    sim.run(base, 15.0);
    CHECK(sim.pulse.hrValid(sim.nowMs()), "removal: no reading before removal");
    sim.run(base, 1.0, false);
    CHECK(!sim.pulse.fingerPresent() && !sim.pulse.hrValid(sim.nowMs()), "removal: stale reading kept");
    std::printf("ok  finger removed -> reading cleared\n");
  }

  // Rate change 70 -> 110 BPM must relock instead of being rejected forever.
  {
    Simulator sim;
    Signal before = base;
    before.bpm = 70;
    sim.beatRate = 70;
    sim.run(before, 15.0);
    Signal after = base;
    after.bpm = 110;
    sim.run(after, 15.0);
    CHECK(sim.pulse.hrValid(sim.nowMs()) && std::fabs(sim.pulse.bpm() - 110) <= 2.5,
          "rate change: bpm %.1f, expected 110", sim.pulse.bpm());
    std::printf("ok  70 -> 110 BPM relocks              bpm %6.1f\n", sim.pulse.bpm());
  }

  // Signal stops (finger still, no pulse): reading expires.
  {
    Simulator sim;
    sim.run(base, 15.0);
    Signal flat = base;
    flat.irPerfusion = 0.0;
    sim.run(flat, 5.0);
    CHECK(!sim.pulse.hrValid(sim.nowMs()), "flat signal: heart rate kept after pulse vanished");
    std::printf("ok  pulse vanishes -> reading expires\n");
  }

  // Sensor clock 3% slow: the measured sample rate keeps BPM correct.
  {
    Simulator sim;
    sim.fs = 97.0;
    sim.pulse.setSampleRate(97.0);
    sim.run(base, 20.0);
    CHECK(std::fabs(sim.pulse.bpm() - 72) <= 2.0, "97 Hz clock: bpm %.1f", sim.pulse.bpm());
    std::printf("ok  97 Hz sensor clock                 bpm %6.1f\n", sim.pulse.bpm());
  }

  // A gap (FIFO overflow / LED change) must not create a fake long interval.
  {
    Simulator sim;
    sim.run(base, 15.0);
    sim.pulse.markGap(1.5f, true);
    sim.run(base, 10.0);
    CHECK(sim.pulse.hrValid(sim.nowMs()) && std::fabs(sim.pulse.bpm() - 72) <= 2.0, "gap: bpm %.1f",
          sim.pulse.bpm());
    std::printf("ok  gap handling                       bpm %6.1f\n", sim.pulse.bpm());
  }

  // Skin contact but no pulse (noise, breathing-rate drift, slow wander): never a heart rate.
  {
    int validSamples = 0;
    for (unsigned seed = 1; seed <= 5; seed++) {
      Simulator sim;
      sim.random.seed(seed);
      Signal none = base;
      none.irPerfusion = 0.0;
      none.noise = 40;
      none.wander = 400;
      for (int s = 0; s < 6000; s++) {  // 60 s, with an extra 0.3 Hz breathing-like wave
        double t = s / 100.0;
        sim.sample++;
        double breathing = 120 * std::sin(2 * M_PI * 0.3 * t);
        std::normal_distribution<double> n(0, none.noise);
        double ir = none.irDc + breathing + none.wander * std::sin(2 * M_PI * 0.07 * t) + n(sim.random);
        double red = none.redDc + 1.5 * breathing + n(sim.random);
        sim.pulse.addSample((uint32_t)red, (uint32_t)ir, sim.nowMs());
        if (sim.pulse.hrValid(sim.nowMs())) validSamples++;
      }
    }
    CHECK(validSamples == 0, "no pulse: reported a heart rate for %d samples", validSamples);
    std::printf("ok  contact without pulse (5 x 60 s) -> no heart rate\n");
  }

  // SpO2 diagnostics expose the last beat's ratio and fit.
  {
    Simulator sim;
    sim.run(base, 15.0);
    float ratio = sim.pulse.lastRatio(), fit = sim.pulse.lastFit();
    CHECK(std::fabs(ratio - base.ratio) < 0.02, "diagnostics: ratio %.3f", ratio);
    CHECK(fit > 0.95f, "diagnostics: fit %.2f", fit);
    sim.run(base, 1.0, false);
    CHECK(std::isnan(sim.pulse.lastRatio()) && sim.pulse.irRms() == 0.0f, "diagnostics not cleared on removal");
    std::printf("ok  SpO2 diagnostics                   ratio %.3f (true %.3f) fit %.2f\n", ratio, base.ratio, fit);
  }

  // LED controller.
  {
    using ppg::LedController;
    // In range and steady: no change.
    LedController leds;
    bool changed = false;
    for (uint32_t t = 0; t <= 3000; t += 20) changed |= leds.update(t, true, 90000, 120000, false, false);
    CHECK(!changed && leds.ir() == ppg::kLedDefault, "led: changed an in-range steady level");

    // Steady but low: raised once, proportionally.
    LedController low;
    changed = false;
    for (uint32_t t = 0; t <= 1100; t += 20) changed |= low.update(t, true, 25000, 30000, false, false);
    CHECK(changed && low.ir() == 124 && low.red() == 103, "led: low level gave ir 0x%02X red 0x%02X", low.ir(),
          low.red());

    // Level jumping around (contact shifting): no change, however low it gets.
    LedController shaky;
    changed = false;
    for (uint32_t t = 0; t <= 5000; t += 20) {
      float level = (t / 100) % 2 ? 20000.0f : 85000.0f;
      changed |= shaky.update(t, true, level, level, false, false);
    }
    CHECK(!changed, "led: changed while the level was unsteady");

    // Steady but moving: no change.
    LedController moving;
    changed = false;
    for (uint32_t t = 0; t <= 5000; t += 20) changed |= moving.update(t, true, 25000, 25000, true, false);
    CHECK(!changed, "led: changed while moving");

    // Saturated: comes down even if unsteady.
    LedController bright;
    for (uint32_t t = 0; t <= 1100; t += 20) bright.update(t, true, t % 40 ? 262000.0f : 150000.0f, 120000, false, true);
    CHECK(bright.ir() < ppg::kLedDefault, "led: saturated level not reduced (0x%02X)", bright.ir());

    // Contact lost: back to default.
    CHECK(low.update(1200, false, 1000, 1000, false, false) && low.ir() == ppg::kLedDefault &&
              low.red() == ppg::kLedDefault,
          "led: not reset when contact was lost");
    std::printf("ok  LED controller (steady, low, unsteady, moving, saturated, removed)\n");
  }

  if (failures) {
    std::printf("\n%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("\nAll pulse processor checks passed\n");
  return 0;
}
