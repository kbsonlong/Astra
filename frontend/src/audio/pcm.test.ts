import { describe, expect, it } from "vitest";
import { PcmFrameBuffer } from "./pcm";

describe("PcmFrameBuffer", () => {
  it("keeps incomplete PCM until a full frame is available", () => {
    const buffer = new PcmFrameBuffer();

    buffer.append(new Float32Array(100));
    expect(buffer.drain(160)).toEqual([]);
    buffer.append(new Float32Array(60));
    expect(buffer.drain(160)).toHaveLength(1);
  });
});
