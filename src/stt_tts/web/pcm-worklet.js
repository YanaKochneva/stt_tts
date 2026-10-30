// SPDX-License-Identifier: Apache-2.0
class PCMWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sum = 0;
    this.count = 0;
    this.phase = 0;
    this.ratio = sampleRate / 16000;
    this.output = new Int16Array(3200);
    this.used = 0;
    this.energy = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input?.length) return true;
    const channels = input.length;
    for (let i = 0; i < input[0].length; i++) {
      let value = 0;
      for (let channel = 0; channel < channels; channel++) value += input[channel][i] || 0;
      value /= channels;
      this.sum += value;
      this.count++;
      this.phase++;
      if (this.phase >= this.ratio) {
        const sample = Math.max(-1, Math.min(1, this.sum / this.count));
        this.output[this.used++] = Math.round(sample * 32767);
        this.energy += sample * sample;
        this.phase -= this.ratio;
        this.sum = 0;
        this.count = 0;
        if (this.used === this.output.length) {
          const chunk = this.output;
          this.port.postMessage({pcm: chunk.buffer, rms: Math.sqrt(this.energy / this.used)}, [chunk.buffer]);
          this.output = new Int16Array(3200);
          this.used = 0;
          this.energy = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('pcm-worklet', PCMWorklet);
