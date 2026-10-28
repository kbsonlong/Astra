import { describe, expect, it, vi, afterEach } from "vitest";
import {
  deleteMeetingPromptTemplate,
  listMeetingPromptTemplates,
  saveMeetingPromptTemplate,
} from "./prompts";

function response(body: unknown, ok = true): Response {
  return { ok, json: async () => body } as Response;
}

afterEach(() => vi.restoreAllMocks());

describe("meeting prompt API", () => {
  it("loads templates and preserves the default id", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response({ default: "standard", items: [{ id: "standard" }] })));

    await expect(listMeetingPromptTemplates()).resolves.toEqual({
      defaultId: "standard",
      items: [{ id: "standard" }],
    });
  });

  it("sends the editor using POST for new templates", async () => {
    const fetchMock = vi.fn().mockResolvedValue(response({ item: { id: "custom" } }));
    vi.stubGlobal("fetch", fetchMock);

    await saveMeetingPromptTemplate({
      name: "自定义",
      description: "描述",
      chunk_system_prompt: "chunk",
      merge_system_prompt: "merge",
    });

    expect(fetchMock).toHaveBeenCalledWith("/api/meeting/prompt-templates", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        name: "自定义",
        description: "描述",
        chunk_system_prompt: "chunk",
        merge_system_prompt: "merge",
      }),
    }));
  });

  it("encodes ids when deleting templates", async () => {
    const fetchMock = vi.fn().mockResolvedValue(response({}));
    vi.stubGlobal("fetch", fetchMock);

    await deleteMeetingPromptTemplate("custom/template");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/meeting/prompt-templates/custom%2Ftemplate",
      { method: "DELETE" },
    );
  });
});
