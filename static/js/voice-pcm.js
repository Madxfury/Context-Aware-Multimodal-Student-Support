// Sarvam expects 16 kHz mono PCM. This node receives mono at the context rate.
class VoicePcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = new Int16Array(1600);
    this.used = 0;
    this.port.onmessage = ({ data }) => {
      if (data !== 'flush') return;
      this.sendFrame();
      this.port.postMessage('flushed');
    };
  }

  sendFrame() {
    if (!this.used) return;
    const frame = this.frame.slice(0, this.used).buffer;
    this.port.postMessage(frame, [frame]);
    this.used = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    for (const sample of channel) {
      const clamped = Math.max(-1, Math.min(1, sample));
      this.frame[this.used++] = clamped * (clamped < 0 ? 32768 : 32767);
      if (this.used === this.frame.length) this.sendFrame();
    }
    return true;
  }
}

registerProcessor('voice-pcm', VoicePcmProcessor);
