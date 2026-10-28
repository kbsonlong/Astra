export type MeetingPromptTemplate = {
  id: string;
  name: string;
  description: string;
  chunk_system_prompt: string;
  merge_system_prompt: string;
  builtin?: boolean;
  editable?: boolean;
};

export type PromptEditorState = {
  id?: string;
  name: string;
  description: string;
  chunk_system_prompt: string;
  merge_system_prompt: string;
};

type PromptTemplatePayload = {
  default?: string;
  items?: MeetingPromptTemplate[];
  detail?: string;
};

type PromptTemplateItemPayload = {
  detail?: string;
  item?: MeetingPromptTemplate;
};

async function readJson<T>(response: Response): Promise<T> {
  return await response.json() as T;
}

export async function listMeetingPromptTemplates(): Promise<{
  defaultId?: string;
  items: MeetingPromptTemplate[];
}> {
  const response = await fetch("/api/meeting/prompt-templates");
  const payload = await readJson<PromptTemplatePayload>(response);
  if (!response.ok) {
    throw new Error(payload.detail ?? "无法读取纪要模板");
  }
  return { defaultId: payload.default, items: payload.items ?? [] };
}

export async function saveMeetingPromptTemplate(editor: PromptEditorState): Promise<MeetingPromptTemplate> {
  const response = await fetch(
    editor.id
      ? `/api/meeting/prompt-templates/${encodeURIComponent(editor.id)}`
      : "/api/meeting/prompt-templates",
    {
      method: editor.id ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: editor.name,
        description: editor.description,
        chunk_system_prompt: editor.chunk_system_prompt,
        merge_system_prompt: editor.merge_system_prompt,
      }),
    },
  );
  const payload = await readJson<PromptTemplateItemPayload>(response);
  if (!response.ok || !payload.item) {
    throw new Error(payload.detail ?? "模板保存失败");
  }
  return payload.item;
}

export async function deleteMeetingPromptTemplate(id: string): Promise<void> {
  const response = await fetch(`/api/meeting/prompt-templates/${encodeURIComponent(id)}`, { method: "DELETE" });
  const payload = await readJson<{ detail?: string }>(response);
  if (!response.ok) {
    throw new Error(payload.detail ?? "模板删除失败");
  }
}
