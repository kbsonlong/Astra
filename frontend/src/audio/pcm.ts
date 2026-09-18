export const JAEC_SAMPLE_RATE = 16_000;
export const PCM_FRAME_SAMPLES = 160;
const PCM_FRAME_MAGIC = "ASTR";
const PCM_FRAME_VERSION = 1;
const PCM_FRAME_HEADER_BYTES = 14;

export class PcmFrameBuffer {
  // 用一个可增长缓冲 + 读游标(head)避免每次 append/drain 都全量复制。
  // 仅当剩余数据前移收益明显时才压缩,均摊 O(1)。
  private buffer = new Float32Array(0);
  private head = 0;
  private tail = 0;

  private get size(): number {
    return this.tail - this.head;
  }

  append(samples: Float32Array): void {
    if (samples.length === 0) return;
    const free = this.buffer.length - this.tail;
    if (free < samples.length) {
      // 需要扩容或压缩:先把未读数据前移,不够再翻倍扩容。
      const needed = this.size + samples.length;
      if (needed <= this.buffer.length) {
        this.buffer.copyWithin(0, this.head, this.tail);
      } else {
        let capacity = Math.max(this.buffer.length * 2, 1024);
        while (capacity < needed) capacity *= 2;
        const next = new Float32Array(capacity);
        next.set(this.buffer.subarray(this.head, this.tail));
        this.buffer = next;
      }
      this.tail = this.size;
      this.head = 0;
    }
    this.buffer.set(samples, this.tail);
    this.tail += samples.length;
  }

  drain(frameSamples = PCM_FRAME_SAMPLES): Float32Array[] {
    const frames: Float32Array[] = [];
    while (this.size >= frameSamples) {
      // slice 复制出定长帧(下游会持有/编码),游标前进而非搬移整段。
      frames.push(this.buffer.slice(this.head, this.head + frameSamples));
      this.head += frameSamples;
    }
    // 读空后重置游标,避免 buffer 无限增长。
    if (this.head === this.tail) {
      this.head = 0;
      this.tail = 0;
    }
    return frames;
  }

  clear(): void {
    this.buffer = new Float32Array(0);
    this.head = 0;
    this.tail = 0;
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

export function encodePcm16Frame(
  samples: Float32Array,
  sequence: number,
  channel: "microphone" | "reference",
  sampleRate = JAEC_SAMPLE_RATE,
): ArrayBuffer {
  const buffer = new ArrayBuffer(PCM_FRAME_HEADER_BYTES + samples.length * 2);
  const view = new DataView(buffer);
  for (let index = 0; index < PCM_FRAME_MAGIC.length; index += 1) {
    view.setUint8(index, PCM_FRAME_MAGIC.charCodeAt(index));
  }
  view.setUint8(4, PCM_FRAME_VERSION);
  view.setUint8(5, channel === "microphone" ? 0 : 1);
  view.setUint32(6, sequence >>> 0, true);
  view.setUint16(10, sampleRate, true);
  view.setUint16(12, samples.length, true);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(
      PCM_FRAME_HEADER_BYTES + index * 2,
      sample < 0 ? sample * 0x8000 : sample * 0x7fff,
      true,
    );
  }
  return buffer;
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
