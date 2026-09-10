export const JAEC_SAMPLE_RATE = 16_000;

export class PcmCollector {
  private readonly chunks: Float32Array[] = [];
  private length = 0;

  append(samples: Float32Array): void {
    const copy = new Float32Array(samples);
    this.chunks.push(copy);
    this.length += copy.length;
  }

  clear(): void {
    this.chunks.length = 0;
    this.length = 0;
  }

  toFloat32(): Float32Array {
    const output = new Float32Array(this.length);
    let offset = 0;
    for (const chunk of this.chunks) {
      output.set(chunk, offset);
      offset += chunk.length;
    }
    return output;
  }

  toSampleRate(sourceRate: number, targetRate = JAEC_SAMPLE_RATE): Float32Array {
    return resamplePcm(this.toFloat32(), sourceRate, targetRate);
  }
}

export function resamplePcm(
  samples: Float32Array,
  sourceRate: number,
  targetRate: number,
): Float32Array {
  if (sourceRate === targetRate) return new Float32Array(samples);
  if (samples.length === 0) return new Float32Array();
  const outputLength = Math.max(1, Math.round(samples.length * targetRate / sourceRate));
  const output = new Float32Array(outputLength);
  const ratio = sourceRate / targetRate;
  for (let index = 0; index < output.length; index += 1) {
    const position = index * ratio;
    const lower = Math.floor(position);
    const upper = Math.min(lower + 1, samples.length - 1);
    const weight = position - lower;
    output[index] = samples[lower] * (1 - weight) + samples[upper] * weight;
  }
  return output;
}

export function fitPcmLength(samples: Float32Array, targetLength: number): Float32Array {
  const output = new Float32Array(targetLength);
  output.set(samples.subarray(0, targetLength));
  return output;
}

export function fitPcmLengthFromEnd(samples: Float32Array, targetLength: number): Float32Array {
  const output = new Float32Array(targetLength);
  output.set(samples.subarray(Math.max(0, samples.length - targetLength)));
  return output;
}

export function encodePcm16Wav(samples: Float32Array, sampleRate = JAEC_SAMPLE_RATE): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(44 + index * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return buffer;
}
