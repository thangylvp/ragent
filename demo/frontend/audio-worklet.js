/* Resample the microphone to 16 kHz mono and emit fixed 20 ms PCM16LE packets. */
class Pcm16Worklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.source = [];
    this.position = 0;
    this.pcm = [];
    this.packetSamples = 320;
    this.packetCount = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || !channel.length) return true;
    for (let i = 0; i < channel.length; i += 1) this.source.push(channel[i]);

    while (this.position + 1 < this.source.length) {
      const left = Math.floor(this.position);
      const fraction = this.position - left;
      const sample = this.source[left] * (1 - fraction) + this.source[left + 1] * fraction;
      this.pcm.push(Math.max(-1, Math.min(1, sample)));
      this.position += this.ratio;
    }
    // Retain one source sample across render quanta so interpolation remains
    // continuous. Dropping past the current buffer would reset phase and make
    // a 48 kHz microphone drift above the requested 16 kHz output rate.
    const drop = Math.min(
      Math.floor(this.position),
      Math.max(0, this.source.length - 1),
    );
    if (drop) {
      this.source.splice(0, drop);
      this.position -= drop;
    }

    while (this.pcm.length >= this.packetSamples) {
      const packet = this.pcm.splice(0, this.packetSamples);
      const buffer = new ArrayBuffer(packet.length * 2);
      const view = new DataView(buffer);
      let squareSum = 0;
      for (let i = 0; i < packet.length; i += 1) {
        const value = packet[i];
        squareSum += value * value;
        view.setInt16(i * 2, value < 0 ? value * 32768 : value * 32767, true);
      }
      this.port.postMessage({ type: "pcm", buffer }, [buffer]);
      this.packetCount += 1;
      if (this.packetCount % 5 === 0) {
        const rms = Math.sqrt(squareSum / packet.length);
        const dbfs = rms > 0 ? Math.max(-120, 20 * Math.log10(rms)) : -120;
        this.port.postMessage({ type: "level", dbfs });
      }
    }
    return true;
  }
}

registerProcessor("pcm16-worklet", Pcm16Worklet);
