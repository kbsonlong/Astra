# Astra TTS 音色库数据库与 API 设计

更新时间：2026-10-01

本文是 TTS 音色库的实现前设计，不修改代码和数据库。目标是为 IndexTTS 2.5 MLX、
CosyVoice 及未来其他克隆型 TTS 提供统一的本地参考音频、授权、版本和试听管理能力。

## 1. 设计决策

- TTS 音色库独立于会议 `SpeakerProfileStore`。会议声纹用于“识别是谁”，TTS 音色用于“以谁的声音生成”。
- 数据库和音频目录独立于 `tasks.sqlite3`、`speakers.sqlite3`，降低迁移和备份耦合。
- 音频资产不可覆盖；替换参考音频或修改合成默认参数都会创建新的 `voice_revision`。
- 新合成必须引用明确的 `voice_id + voice_revision + reference_sha256` 快照。
- 未确认授权的音色可以上传为草稿，但不能激活、试听合成或用于实时 TTS。
- 归档和撤销是可追溯状态变化，不提供默认硬删除；历史任务引用的资产不被删除。
- 原始上传和规范化 WAV 都放在非 Web 根目录，仅通过受鉴权 API 下载。
- 所有音色 API 复用 Astra 现有管理员会话中间件；不新增另一套 token。

## 2. 数据目录与配置

建议增加以下配置，默认不影响现有 Piper/MLX Audio/CosyVoice：

```text
TTS_VOICE_STORE_PATH=~/.astra/tts-voices.sqlite3
TTS_VOICE_AUDIO_DIR=~/.astra/tts-voices
TTS_VOICE_MIN_REFERENCE_SECONDS=5
TTS_VOICE_MAX_REFERENCE_SECONDS=15
TTS_VOICE_MAX_UPLOAD_BYTES=20971520
TTS_VOICE_DEFAULT_PAGE_SIZE=50
```

目录布局：

```text
~/.astra/
  tts-voices.sqlite3
  tts-voices/
    <voice_id>/
      rev-0001-original.<suffix>
      rev-0001-reference.wav
      rev-0002-original.<suffix>
      rev-0002-reference.wav
```

数据库只保存相对路径，例如 `voice-id/rev-0001-reference.wav`，不保存可变的绝对路径。
读取文件时必须解析后确认路径仍位于 `TTS_VOICE_AUDIO_DIR` 内，拒绝 `..`、符号链接和目录穿越。

## 3. 数据模型

### 3.1 `tts_voices`：音色身份和当前状态

```sql
CREATE TABLE IF NOT EXISTS tts_voices (
    voice_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'active', 'archived', 'revoked')),
    current_revision INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at TEXT,
    revoked_at TEXT,
    CHECK (length(trim(display_name)) > 0)
);

CREATE INDEX IF NOT EXISTS idx_tts_voices_status_updated
    ON tts_voices(status, updated_at DESC);
```

`voice_id` 使用 UUID；`current_revision` 指向当前可用于新请求的 revision。`draft` 可以没有
current revision，`active` 必须有一个已校验且授权确认的 revision。

### 3.2 `tts_voice_revisions`：不可变参考资产和合成默认值

```sql
CREATE TABLE IF NOT EXISTS tts_voice_revisions (
    voice_id TEXT NOT NULL REFERENCES tts_voices(voice_id),
    revision INTEGER NOT NULL,
    lifecycle TEXT NOT NULL DEFAULT 'candidate'
        CHECK (lifecycle IN ('candidate', 'ready', 'superseded', 'revoked')),
    backend_family TEXT NOT NULL DEFAULT 'indextts25_mlx',
    reference_sha256 TEXT NOT NULL,
    original_sha256 TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    original_relpath TEXT NOT NULL,
    reference_relpath TEXT NOT NULL,
    mime_type TEXT NOT NULL DEFAULT 'audio/wav',
    language TEXT NOT NULL DEFAULT 'zh',
    duration_s REAL NOT NULL,
    speech_duration_s REAL,
    sample_rate INTEGER NOT NULL,
    channels INTEGER NOT NULL,
    sample_width_bytes INTEGER NOT NULL,
    quality_score REAL,
    default_params_json TEXT NOT NULL DEFAULT '{}',
    consent_status TEXT NOT NULL DEFAULT 'unknown'
        CHECK (consent_status IN ('unknown', 'confirmed', 'revoked')),
    consent_note TEXT,
    consent_confirmed_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    superseded_at TEXT,
    PRIMARY KEY (voice_id, revision),
    CHECK (duration_s > 0),
    CHECK (sample_rate > 0),
    CHECK (channels > 0),
    CHECK (sample_width_bytes > 0),
    CHECK (json_valid(default_params_json)),
    CHECK (consent_status != 'confirmed' OR consent_confirmed_at IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_tts_voice_revisions_sha
    ON tts_voice_revisions(reference_sha256);
CREATE INDEX IF NOT EXISTS idx_tts_voice_revisions_backend
    ON tts_voice_revisions(backend_family, lifecycle);
```

字段规则：

- `original_sha256` 是用户上传字节的哈希；`reference_sha256` 是实际送入模型的规范化 WAV 哈希。
- `reference_relpath` 指向稳定的规范化音频；模型永远不读取用户原始扩展名文件。
- `default_params_json` 只保存白名单参数，例如 `language`、`greedy`、`seed`、`duration_factor`，不保存任意代码或路径。
- `backend_family` 表示适配能力，不等于具体模型版本；具体 `model_revision` 在任务快照中记录。
- 授权状态放在 revision 上，因为替换参考音频后授权范围可能不同。

### 3.3 可选 `tts_voice_events`：审计和撤销记录

MVP 可以先写应用日志；若需要 UI 审计或合规导出，增加：

```sql
CREATE TABLE IF NOT EXISTS tts_voice_events (
    event_id TEXT PRIMARY KEY,
    voice_id TEXT NOT NULL REFERENCES tts_voices(voice_id),
    revision INTEGER,
    event_type TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (json_valid(detail_json))
);

CREATE INDEX IF NOT EXISTS idx_tts_voice_events_voice_time
    ON tts_voice_events(voice_id, created_at DESC);
```

审计详情只保存动作、旧/新状态、revision 和安全的备注，不保存 token、完整本地路径或参考文本。

## 4. 状态和版本规则

```text
上传音频
   |
   v
 draft + candidate + consent=unknown
   | 通过格式/时长/质量校验，并确认授权
   v
 active + ready + consent=confirmed
   | 新 revision
   v
旧 revision=superseded，当前 revision=ready

active --归档--> archived
active/draft/archived --撤销授权--> revoked
```

规则：

1. 只有 `tts_voices.status=active`、current revision 为 `ready`、`consent_status=confirmed` 才可合成。
2. `archived` 禁止新合成，但允许读取历史元数据；是否允许试听由产品策略决定，默认禁止。
3. `revoked` 立即阻断新合成和音频下载；数据库和文件保留，供审计和历史任务解释。
4. 上传替换音频时先写临时文件并完成校验，再在一个 `BEGIN IMMEDIATE` 事务内创建 revision 和切换 current revision。
5. 任何合成任务在提交时复制 revision 快照；之后归档或撤销不改变已提交任务的审计数据。

## 5. 音频入库流程

```text
multipart upload
      |
      +-- 鉴权、扩展名和大小限制
      +-- 临时文件落盘
      +-- afconvert/soundfile 规范化为 WAV PCM16 mono
      +-- 校验 5–15s、可解码、非空、单人/质量条件
      +-- 计算 original_sha256/reference_sha256
      +-- 写音频文件到 voice/revision 临时路径
      +-- SQLite 事务写 voice/revision
      +-- 原子 rename 到最终路径
      v
draft revision
```

推荐规范化格式为 16kHz、单声道、16-bit PCM WAV；具体采样率应与目标后端实测要求保持一致，
不能在 API 层假设模型输出采样率等于参考输入采样率。当前 IndexTTS MLX 参考音频限制不超过 15 秒，
因此服务端拒绝超长音频，而不是截断用户文件。

质量校验最小集：

- 文件可解码，声道为 1；
- 时长在配置范围内；
- PCM 非全静音，峰值和 RMS 不越界；
- 可选 VAD 得到 `speech_duration_s` 和质量分；
- 不把多说话人判定当作绝对安全结论，授权仍必须人工确认。

失败清理规则：数据库事务失败时删除临时/最终文件；数据库成功但后续清理失败时记录 orphan，
由启动清理任务或显式维护命令回收，不能直接删除数据库记录掩盖不一致。

## 6. Store 设计

新增 `backend/app/core/tts_voice_store.py`，参考现有 Store 的 SQLite 模式，但不要修改
`SpeakerProfileStore`。建议职责：

```text
TtsVoiceStore
  create_draft(...)
  add_revision(...)
  get(voice_id, include_archived=False)
  list(..., page, page_size)
  update_metadata(...)
  confirm_consent(...)
  revoke_consent(...)
  activate(...)
  archive(...)
  current_revision(voice_id)
  get_revision(voice_id, revision)
  resolve_audio_path(voice_id, revision, kind)
  create_synthesis_snapshot(...)
```

实现约束：

- 每次连接启用 `PRAGMA foreign_keys = ON`；schema 初始化必须幂等。
- 写 revision、切换 current revision、状态迁移使用 `BEGIN IMMEDIATE`。
- list/get 只返回 dataclass 或脱敏 DTO，不把绝对路径返回到 API。
- `resolve_audio_path()` 必须校验数据库相对路径、根目录、文件存在性和普通文件属性。
- 需要 schema 版本时增加 `tts_voice_schema_meta`，不要依赖隐式 `CREATE TABLE IF NOT EXISTS` 修改旧列。

## 7. API 设计

所有 `/api/tts/voices` 路由自动受到现有管理员会话中间件保护。

### 7.1 创建草稿并上传第一版

```http
POST /api/tts/voices
Content-Type: multipart/form-data

name=小雅
language=zh
backend_family=indextts25_mlx
file=<reference.wav>
```

返回 `201`：

```json
{
  "voice_id": "uuid",
  "status": "draft",
  "current_revision": 1,
  "revision": {
    "revision": 1,
    "lifecycle": "candidate",
    "consent_status": "unknown",
    "duration_s": 8.4,
    "speech_duration_s": 7.9,
    "sample_rate": 16000,
    "channels": 1,
    "backend_family": "indextts25_mlx",
    "reference_sha256": "..."
  }
}
```

上传只创建草稿，不允许客户端直接把 `consent_status` 伪造为 `confirmed`。

### 7.2 查询和详情

```text
GET /api/tts/voices?status=active&language=zh&backend_family=indextts25_mlx&q=小雅&page=1&page_size=50
GET /api/tts/voices/{voice_id}
GET /api/tts/voices/{voice_id}/revisions
GET /api/tts/voices/{voice_id}/revisions/{revision}
```

列表返回分页信息和当前 revision 摘要；详情返回授权状态、质量、默认参数和音频 URL，但不返回
服务器绝对路径、token 或参考文本。未授权的音色只对管理员可见，且不能进入合成选择器。

### 7.3 修改元数据和创建新 revision

```http
PATCH /api/tts/voices/{voice_id}
Content-Type: application/json

{
  "display_name": "小雅-会议版"
}
```

普通 `PATCH` 只修改显示名等不影响合成结果的元数据，不产生 revision；修改参考音频或合成默认参数必须使用：

```http
POST /api/tts/voices/{voice_id}/revisions
Content-Type: multipart/form-data

file=<new-reference.wav>
backend_family=indextts25_mlx
language=zh
default_params={"duration_factor":1.0}
```

新 revision 初始为 `candidate`、授权状态 `unknown`，不会自动替换 active revision。

### 7.4 授权、激活、归档和撤销

```text
POST /api/tts/voices/{voice_id}/revisions/{revision}/consent
POST /api/tts/voices/{voice_id}/revisions/{revision}/revoke-consent
POST /api/tts/voices/{voice_id}/activate
POST /api/tts/voices/{voice_id}/archive
POST /api/tts/voices/{voice_id}/restore
```

确认授权请求示例：

```json
{
  "confirmation": "本人已授权 Astra 在本机进行语音合成",
  "note": "2026-10-01 线下确认"
}
```

服务端只保存确认时间和备注，不保存不必要的身份证明材料。`confirmation` 必须匹配后端要求的
明确确认短语，不能用 `true` 绕过。激活操作检查音频、质量、授权、backend family 和 revision
完整性；撤销授权立即使当前 voice 不可用于新合成。

### 7.5 试听参考音频

```text
GET /api/tts/voices/{voice_id}/audio
GET /api/tts/voices/{voice_id}/revisions/{revision}/audio
```

返回 `audio/wav`，通过数据库解析路径后使用 `FileResponse`。禁止把音频目录挂成公开静态目录。
归档/撤销后的访问策略默认返回 `404` 或 `409`，不要让已撤销音色继续被 UI 误用。

### 7.6 使用指定音色进行试听合成

```http
POST /api/tts/voices/{voice_id}/test
Content-Type: application/json

{
  "revision": 3,
  "text": "你好，这是 Astra 的本地音色测试。",
  "language": "zh",
  "params": {
    "greedy": true
  }
}
```

返回 `audio/wav`，响应头附带：

```text
X-Astra-Voice-Id: <uuid>
X-Astra-Voice-Revision: 3
X-Astra-Reference-SHA256: <sha256>
X-Astra-Backend: indextts_mlx
X-Astra-Model-Revision: <revision>
```

试听请求不得自动修改 current revision，也不得接受任意本地路径。请求只允许使用已激活且授权
确认的 revision；文本长度和生成参数必须受限，避免把试听接口变成无界后台任务。

## 8. API 错误与并发语义

| HTTP | 场景 |
| --- | --- |
| 400 | 参数格式、状态转换或生成参数不合法 |
| 401 | 管理员会话缺失，由现有 middleware 返回 |
| 404 | voice/revision/audio 不存在，或已撤销资源不可见 |
| 409 | revision 冲突、状态不允许、SHA/文件不一致、正在切换 |
| 413 | 上传超过 `TTS_VOICE_MAX_UPLOAD_BYTES` |
| 415 | 不支持的媒体类型 |
| 422 | 音频不可解码、时长/声道/静音/质量校验失败 |
| 429 | IndexTTS worker 队列或试听并发上限 |
| 503 | 后端未安装、模型不可用或 TTS 引擎暂时不可用 |

写操作使用 SQLite 事务；更新 current revision 时必须检查旧 revision，避免两个管理员请求互相覆盖。
试听和实时 TTS 读取一次 immutable snapshot 后再进入 adapter，不在推理期间持有 SQLite 事务。

## 9. 与 IndexTTS 适配器的连接

适配器收到的不是裸路径，而是经过 Store 校验的快照：

```json
{
  "voice_id": "uuid",
  "voice_revision": 3,
  "reference_audio_path": "<resolved private path>",
  "reference_sha256": "...",
  "backend_family": "indextts25_mlx",
  "model_revision": "<pinned model revision>",
  "language": "zh",
  "generation": {
    "greedy": true,
    "duration_factor": 1.0
  }
}
```

适配器只在 worker 内根据 `(voice_id, voice_revision, reference_sha256)` 缓存 `SpeakerContext`。
Store 的 `archive`、`revoke` 或新 revision 不需要主动操作 MLX 内存；adapter 在下一次请求发现
snapshot 不再有效时丢弃对应缓存即可。

## 10. 任务快照与历史可复现

实时会话或异步任务至少保存：

```json
{
  "tts": {
    "backend": "indextts_mlx",
    "voice_id": "uuid",
    "voice_revision": 3,
    "reference_sha256": "...",
    "model_revision": "...",
    "language": "zh",
    "generation_params": {"greedy": true}
  }
}
```

禁止只保存 `reference_audio_path` 或当前 voice 名称。用户改名、归档、撤销或替换音频后，历史
任务仍应能显示当时使用的音色版本；是否允许再次重放由任务权限和撤销策略单独决定。

## 11. 测试与验收

### Store 单元测试

1. 创建草稿、创建 revision、查询 current revision；
2. 重复 revision 号被拒绝，SHA 和相对路径被保存；
3. 未授权不能激活或试听；
4. 归档、恢复、撤销的状态转换符合状态机；
5. 新 revision 不覆盖旧文件和旧元数据；
6. 并发切换 current revision 不丢更新；
7. 路径穿越、符号链接和数据库外文件被拒绝；
8. 事务失败时不留下可被 API 访问的孤儿文件。

### API 测试

1. 未登录访问全部 `/api/tts/voices` 路由返回 401；
2. 上传大小、扩展名、时长、声道和坏文件返回正确状态码；
3. 列表分页、搜索、状态过滤和脱敏字段正确；
4. 音频下载只返回授权/允许状态的文件；
5. `/test` 固定引用指定 revision，不读取 current revision 的变化；
6. `/test` 输出包含 voice/model revision 响应头；
7. 撤销授权后新试听立即失败，旧任务快照仍可查询；
8. 试听并发超过限制返回 429，不阻塞 API 事件循环。

### 与 IndexTTS MLX 的集成验收

- 使用授权且 5–15 秒参考 WAV 创建 active voice；
- L2 短句合成成功，输出 22050Hz 单声道 WAV；
- 同一 revision 连续 3 次复用 `SpeakerContext`；
- 新 revision 与旧 revision 的音色引用不串；
- `/api/health` 能分别展示音色库配置状态和 IndexTTS 模型状态；
- Piper 回退、WebSocket generation 打断和应用重启后数据库恢复通过。

## 12. 分阶段实现

### V0：数据库和 Store

- 新增配置、`TtsVoiceStore`、schema 初始化和路径安全函数；
- 完成音频临时文件、规范化、哈希、时长和事务写入；
- 不接入 IndexTTS 推理。

### V1：管理 API 和参考音频试听

- 新增 schemas、`/api/tts/voices` 路由、上传/列表/详情/音频下载；
- 实现授权、激活、归档、撤销和 revision；
- 完成 API/Store 测试。

### V2：IndexTTS 试听与会话选择

- 将 immutable voice snapshot 接入 `IndexTTS25MlxClient`；
- `/test` 和 WebSocket 传递 `voice_id`/revision；
- 记录任务快照和模型 revision；
- 真机 L2/L3 通过后才开放为用户可选后端。

## 13. 完成标准

- 音色数据与会议声纹数据完全分离；
- 参考音频、授权、SHA、revision 和模型版本可追溯；
- 无 API 路径穿越、公开静态目录或未授权试听；
- 音频替换不会覆盖旧资产，历史快照不依赖 current revision；
- SQLite 写入、文件写入和失败清理边界明确；
- API 可分页、筛选、归档、撤销、试听和下载；
- 设计可被 IndexTTS MLX、CosyVoice 等多个后端复用；
- 真实 IndexTTS 模型尚未完成 L2 前，音色库和适配器保持 opt-in。
