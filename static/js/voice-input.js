class ChatVoiceInput {
  constructor(onPreview, onStop) {
    this.onPreview = onPreview;
    this.onStop = onStop;
    this.controller = new AbortController();
    this.finalLines = [];
    this.recordedChunks = [];
    this.waitingFrames = [];
  }

  async start() {
    const signal = this.controller.signal;
    if (window.AudioContext) {
      try {
        // Create and resume in the click handler for browsers with autoplay policies.
        this.context = new AudioContext({ sampleRate: 16000 });
        this.context.resume().catch(() => {});
      } catch {
        // MediaRecorder provides the recorded-note fallback.
      }
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });
    if (signal.aborted) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    this.stream = stream;
    if (window.MediaRecorder) {
      this.recorder = new MediaRecorder(stream);
      this.recording = new Promise(resolve => {
        this.recorder.addEventListener('dataavailable', event => {
          if (event.data.size) this.recordedChunks.push(event.data);
        });
        this.recorder.addEventListener('stop', () => {
          resolve(new Blob(this.recordedChunks, { type: this.recorder.mimeType }));
          this.onStop();
        }, { once: true });
        this.recorder.addEventListener('error', () => {
          resolve(null);
          this.onStop();
        }, { once: true });
      });
      this.recorder.start(100);
    }
    stream.getAudioTracks().forEach(track => track.addEventListener('ended', this.onStop, { once: true }));
    this.startedAt = performance.now();
    this.limit = setTimeout(this.onStop, 60000);
    if (this.context) {
      this.source = this.context.createMediaStreamSource(stream);
      this.watchForSilence();
    }
    this.startLive();
  }

  async startLive() {
    if (!window.WebSocket || this.context?.sampleRate !== 16000 || !this.context.audioWorklet) {
      this.liveUnavailable = true;
      return;
    }
    try {
      await this.context.audioWorklet.addModule('/static/js/voice-pcm.js');
      if (this.controller.signal.aborted || this.stopping) return;
      const socketURL = new URL('/transcribe/live', location.href);
      socketURL.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const socket = new WebSocket(socketURL);
      this.socket = socket;
      this.liveResult = new Promise(resolve => {
        this.resolveLive = resolve;
        socket.addEventListener('open', () => {
          for (const frame of this.waitingFrames) socket.send(frame);
          this.waitingFrames = [];
        });
        socket.addEventListener('message', ({ data }) => {
          const event = JSON.parse(data);
          const finalHasWords = event.type === 'final' && /[\p{L}\p{N}]/u.test(event.text);
          if (finalHasWords) {
            this.finalLines.push(event.text.trim());
          }
          if (event.type === 'partial' || event.type === 'final') {
            const partial = event.type === 'partial' && /[\p{L}\p{N}]/u.test(event.text) ? event.text : '';
            this.onPreview([...this.finalLines, partial].filter(Boolean).join(' '));
            if (finalHasWords && event.end_of_turn && !this.stopping) this.onStop();
          } else if (event.type === 'done') {
            resolve(this.finalLines.join(' '));
          } else if (event.type === 'unavailable') {
            socket.close();
            resolve('');
          }
        });
        const useFallback = () => {
          this.liveUnavailable = true;
          resolve('');
        };
        socket.addEventListener('close', useFallback);
        socket.addEventListener('error', useFallback);
      });
      this.worklet = new AudioWorkletNode(this.context, 'voice-pcm', {
        channelCount: 1, channelCountMode: 'explicit'
      });
      this.worklet.port.onmessage = ({ data }) => {
        if (data === 'flushed') {
          this.endLive();
        } else if (!this.liveEnded && socket.readyState === WebSocket.OPEN) {
          socket.send(data);
        } else if (socket.readyState === WebSocket.CONNECTING) {
          if (this.waitingFrames.length < 50) this.waitingFrames.push(data);
          else socket.close();
        }
      };
      this.source.connect(this.worklet).connect(this.context.destination);
    } catch {
      this.liveUnavailable = true;
      this.socket?.close();
    }
  }

  watchForSilence() {
    const analyser = this.context.createAnalyser();
    analyser.fftSize = 2048;
    this.source.connect(analyser);
    const samples = new Float32Array(analyser.fftSize);
    let audibleFrames = 0;
    let lastSoundAt = 0;

    // Use the local level only for the recorded-note fallback.
    this.silenceCheck = setInterval(() => {
      analyser.getFloatTimeDomainData(samples);
      const rms = Math.sqrt(samples.reduce((energy, sample) => energy + sample * sample, 0) / samples.length);
      if (rms >= 0.015) {
        audibleFrames++;
        lastSoundAt = performance.now();
      } else if (this.liveUnavailable && audibleFrames >= 3 && performance.now() - lastSoundAt >= 2000) {
        this.onStop();
      }
    }, 100);
  }

  endLive() {
    if (this.liveEnded) return;
    this.liveEnded = true;
    this.releaseInput();
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send('end');
    else {
      this.socket?.close();
      this.resolveLive?.('');
    }
  }

  async finish() {
    this.stopping = true;
    const elapsed = performance.now() - this.startedAt;
    clearTimeout(this.limit);
    clearInterval(this.silenceCheck);
    if (this.recorder?.state === 'recording') this.recorder.stop();

    let liveDeadline;
    let flushDeadline;
    try {
      if (elapsed < 500) throw new Error('That voice note was too short. Try again when you’re ready.');
      if (this.worklet) {
        this.worklet.port.postMessage('flush');
        flushDeadline = setTimeout(() => this.endLive(), 250);
      }
      const liveTranscript = this.liveResult && await Promise.race([
        this.liveResult,
        new Promise(resolve => { liveDeadline = setTimeout(() => resolve(''), 4500); })
      ]);
      this.releaseMicrophone();
      this.controller.signal.throwIfAborted();
      if (liveTranscript) return liveTranscript;
      const blob = await this.recording;
      if (!blob?.size) throw new Error('No speech was detected. Try again or type a message.');
      const form = new FormData();
      form.append('audio', blob, 'voice-note');
      const response = await fetch('/transcribe', {
        method: 'POST', body: form,
        signal: AbortSignal.any([this.controller.signal, AbortSignal.timeout(25000)])
      });
      const data = await response.json();
      if (!response.ok || !data.transcription) {
        throw new Error(data.error || 'No speech was detected. Try again or type a message.');
      }
      return data.transcription;
    } finally {
      clearTimeout(liveDeadline);
      clearTimeout(flushDeadline);
      this.releaseMicrophone();
    }
  }

  releaseInput() {
    clearInterval(this.silenceCheck);
    if (this.recorder?.state === 'recording') this.recorder.stop();
    this.stream?.getTracks().forEach(track => track.stop());
    this.stream = null;
    const context = this.context;
    this.context = null;
    if (context && context.state !== 'closed') context.close().catch(() => {});
  }

  releaseMicrophone() {
    clearTimeout(this.limit);
    this.releaseInput();
    this.socket?.close();
    this.waitingFrames = [];
  }

  cancel() {
    this.controller.abort();
    this.resolveLive?.('');
    this.releaseMicrophone();
  }
}
