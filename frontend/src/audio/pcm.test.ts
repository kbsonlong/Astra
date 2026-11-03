import { describe, expect, it } from "vitest";
import { PcmFrameBuffer, decodeTtsFrame } from "./pcm";

describe("PcmFrameBuffer", () => {
  it("keeps incomplete PCM until a full frame is available", () => {
    const buffer = new PcmFrameBuffer();

    buffer.append(new Float32Array(100));
    expect(buffer.drain(160)).toEqual([]);
    buffer.append(new Float32Array(60));
    expect(buffer.drain(160)).toHaveLength(1);
  });
});

function buildTtsFrame(
  generationId: number,
  sequence: number,
  sampleRate: number,
  payload: Uint8Array,
): ArrayBuffer {
  const header = new ArrayBuffer(18);
  const view = new DataView(header);
  const magic = "ATTS";
  for (let index = 0; index < magic.length; index += 1) {
    view.setUint8(index, magic.charCodeAt(index));
  }
  view.setUint8(4, 1); // version
  view.setUint8(5, 0); // reserved
  view.setUint32(6, generationId, true);
  view.setUint32(10, sequence, true);
  view.setUint32(14, sampleRate, true);
  const out = new Uint8Array(18 + payload.length);
  out.set(new Uint8Array(header), 0);
  out.set(payload, 18);
  return out.buffer;
}

describe("decodeTtsFrame", () => {
  it("decodes the header fields and payload from a backend-shaped frame", () => {
    const payload = new Uint8Array([1, 2, 3, 4]);
    const frame = decodeTtsFrame(buildTtsFrame(7, 3, 24_000, payload));
    expect(frame).not.toBeNull();
    expect(frame?.generationId).toBe(7);
    expect(frame?.sequence).toBe(3);
    expect(frame?.sampleRate).toBe(24_000);
    expect(new Uint8Array(frame!.payload)).toEqual(payload);
  });

  it("rejects a frame whose magic is not ATTS", () => {
    const buffer = buildTtsFrame(0, 0, 16_000, new Uint8Array());
    new DataView(buffer).setUint8(0, "X".charCodeAt(0));
    expect(decodeTtsFrame(buffer)).toBeNull();
  });
});
