import { useEffect, useState } from "react";

type TrainingConfig = {
  model_path: string;
  train_file: string;
  eval_file: string;
  output_dir: string;
  device: "auto" | "cuda" | "mps" | "cpu";
  precision: "bf16" | "fp16" | "fp32";
  batch_size: number;
  grad_acc: number;
  learning_rate: number;
  epochs: number;
  save_steps: number;
  save_total_limit: number;
  num_workers: number;
  pin_memory: boolean;
  persistent_workers: boolean;
  prefetch_factor: number;
  resume_from: string;
  resume_latest: boolean;
};

type TrainingStatus = {
  status: "idle" | "processing" | "done" | "failed";
  task_id?: string;
  returncode?: number | null;
  log_path?: string;
  output_dir?: string;
};

type ApprovedDataset = {
  id: string;
  task_id: string;
  path: string;
  approved_count: number;
  total_count: number;
};

export default function TrainingPage() {
  const [config, setConfig] = useState<TrainingConfig | null>(null);
  const [status, setStatus] = useState("训练参数未加载");
  const [job, setJob] = useState<TrainingStatus>({ status: "idle" });
  const [datasets, setDatasets] = useState<ApprovedDataset[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    void refreshConfig();
    void refreshStatus();
    void refreshDatasets();
  }, []);

  useEffect(() => {
    if (job.status !== "processing") return undefined;
    const timer = window.setInterval(() => { void refreshStatus(); }, 2000);
    return () => window.clearInterval(timer);
  }, [job.status]);

  async function refreshConfig() {
    try {
      const response = await fetch("/api/training/config");
      const payload = await response.json() as { config?: TrainingConfig; detail?: string };
      if (!response.ok || !payload.config) throw new Error(payload.detail ?? "无法读取训练参数");
      setConfig(payload.config);
      setStatus("已加载训练参数");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "训练参数加载失败");
    }
  }

  async function refreshStatus() {
    try {
      const response = await fetch("/api/training/status");
      const payload = await response.json() as TrainingStatus & { detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? "训练状态读取失败");
      setJob(payload);
      if (payload.status === "processing") setStatus(`训练进行中 · ${payload.task_id ?? ""}`);
      if (payload.status === "done") setStatus("训练完成，适配器已写入输出目录");
      if (payload.status === "failed") setStatus(`训练失败（退出码 ${payload.returncode ?? "未知"}），请查看日志`);
      if (payload.status === "idle") setStatus("当前没有运行中的训练任务");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "训练状态读取失败");
    }
  }

  async function refreshDatasets() {
    try {
      const response = await fetch("/api/training/datasets");
      const payload = await response.json() as { items?: ApprovedDataset[]; detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? "无法加载已审批数据");
      setDatasets(payload.items ?? []);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "已审批数据加载失败");
    }
  }

  async function saveConfig() {
    if (!config) return;
    setBusy(true);
    try {
      const response = await fetch("/api/training/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      const payload = await response.json() as { config?: TrainingConfig; detail?: string };
      if (!response.ok || !payload.config) throw new Error(payload.detail ?? "训练参数保存失败");
      setConfig(payload.config);
      setStatus("训练参数已保存");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "训练参数保存失败");
    } finally {
      setBusy(false);
    }
  }

  async function startTraining() {
    if (!config) return;
    setBusy(true);
    try {
      const response = await fetch("/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      const payload = await response.json() as TrainingStatus & { detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? "训练启动失败");
      setJob(payload);
      setStatus(`训练已启动 · ${payload.task_id ?? ""}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "训练启动失败");
    } finally {
      setBusy(false);
    }
  }

  async function stopTraining() {
    setBusy(true);
    try {
      const response = await fetch("/api/training/stop", { method: "POST" });
      const payload = await response.json() as TrainingStatus & { detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? "训练停止失败");
      setJob(payload);
      setStatus("已请求停止训练");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "训练停止失败");
    } finally {
      setBusy(false);
    }
  }

  function update<K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) {
    setConfig((current) => current ? { ...current, [key]: value } : current);
  }

  function textField(key: "model_path" | "train_file" | "eval_file" | "output_dir" | "resume_from", label: string) {
    if (!config) return null;
    return (
      <div className="field">
        <label className="field__label" htmlFor={`tf-${key}`}>{label}</label>
        <input className="input input--mono" id={`tf-${key}`} value={config[key]} onChange={(event) => update(key, event.target.value)} />
      </div>
    );
  }

  function datasetField(key: "train_file" | "eval_file", label: string) {
    if (!config) return null;
    return (
      <div className="field">
        <label className="field__label" htmlFor={`tf-${key}`}>{label}</label>
        <select className="select" id={`tf-${key}`} value={datasets.some((item) => item.path === config[key]) ? config[key] : ""} onChange={(event) => update(key, event.target.value)}>
          <option value="">{datasets.length ? "选择已审批会议数据" : "暂无已审批数据"}</option>
          {datasets.map((dataset) => (
            <option value={dataset.path} key={dataset.id}>{dataset.task_id} · 已通过 {dataset.approved_count}/{dataset.total_count} 段</option>
          ))}
        </select>
        <span className="field__hint">完成逐段审核后，点击右上角重载即可看到新数据。</span>
      </div>
    );
  }

  function numberField(key: "batch_size" | "grad_acc" | "learning_rate" | "epochs" | "save_steps" | "save_total_limit" | "num_workers" | "prefetch_factor", label: string, step = "1") {
    if (!config) return null;
    return (
      <div className="field">
        <label className="field__label" htmlFor={`tf-${key}`}>{label}</label>
        <input className="input input--mono" type="number" min="0" step={step} id={`tf-${key}`} value={config[key]} onChange={(event) => update(key, Number(event.target.value) as TrainingConfig[typeof key])} />
      </div>
    );
  }

  const jobBadge =
    job.status === "processing" ? "badge--accent" : job.status === "done" ? "badge--success" : job.status === "failed" ? "badge--danger" : "badge--neutral";
  const jobLabel =
    job.status === "processing" ? "训练运行中" : job.status === "done" ? "已完成" : job.status === "failed" ? "失败" : "待机";

  return (
    <section className="view" aria-label="模型训练">
      <div className="page-head">
        <div className="page-head__text">
          <div className="eyebrow">04 · 模型训练</div>
          <h1>用已审核的会议数据微调 ASR</h1>
          <p>只有审核通过的片段才会进入训练数据集。训练在独立进程中执行，不影响正在进行的会议任务。</p>
        </div>
        <div className="page-head__actions">
          <span className={`badge ${jobBadge}`}>
            {job.status === "processing" && <span className="badge__dot" />}
            {jobLabel}
          </span>
        </div>
      </div>

      <div className="train-grid">
        <div className="col">
          <div className="card">
            <div className="card__head">
              <h3>训练配置</h3>
              <button className="btn btn--ghost btn--sm" type="button" onClick={() => void refreshConfig()} disabled={busy}>重载</button>
              <button className="btn btn--secondary btn--sm" type="button" onClick={() => void saveConfig()} disabled={!config || busy}>保存参数</button>
              {job.status === "processing" ? (
                <button className="btn btn--danger btn--sm" type="button" onClick={() => void stopTraining()} disabled={busy}>停止训练</button>
              ) : (
                <button className="btn btn--primary btn--sm" type="button" onClick={() => void startTraining()} disabled={!config || busy}>启动训练</button>
              )}
            </div>
            {config && (
              <div className="form-grid">
                {textField("model_path", "训练基座模型")}
                {datasetField("train_file", "已审批训练数据")}
                {datasetField("eval_file", "已审批验证数据")}
                {textField("output_dir", "输出目录")}
                <div className="field">
                  <label className="field__label" htmlFor="tf-device">设备</label>
                  <select className="select" id="tf-device" value={config.device} onChange={(event) => update("device", event.target.value as TrainingConfig["device"])}>
                    <option value="cuda">CUDA</option>
                    <option value="mps">Apple MPS（实验）</option>
                    <option value="cpu">CPU（不推荐）</option>
                    <option value="auto">自动</option>
                  </select>
                </div>
                <div className="field">
                  <label className="field__label" htmlFor="tf-precision">精度</label>
                  <select className="select" id="tf-precision" value={config.precision} onChange={(event) => update("precision", event.target.value as TrainingConfig["precision"])}>
                    <option value="bf16">BF16</option>
                    <option value="fp16">FP16</option>
                    <option value="fp32">FP32</option>
                  </select>
                </div>
                {numberField("batch_size", "Batch size")}
                {numberField("grad_acc", "梯度累积")}
                {numberField("learning_rate", "学习率", "0.000001")}
                {numberField("epochs", "Epochs")}
                {numberField("save_steps", "保存步数")}
                {numberField("save_total_limit", "保留 checkpoint")}
                {numberField("num_workers", "数据线程")}
                {numberField("prefetch_factor", "预取因子")}
                {textField("resume_from", "恢复 checkpoint（可选）")}
              </div>
            )}
            {config && (
              <>
                <div className="divider" />
                <div className="checks">
                  <label className="check"><input type="checkbox" checked={config.pin_memory} onChange={(event) => update("pin_memory", event.target.checked)} />启用 pin memory</label>
                  <label className="check"><input type="checkbox" checked={config.persistent_workers} onChange={(event) => update("persistent_workers", event.target.checked)} />保持数据线程</label>
                  <label className="check"><input type="checkbox" checked={config.resume_latest} onChange={(event) => update("resume_latest", event.target.checked)} />自动恢复最新 checkpoint</label>
                </div>
              </>
            )}
          </div>

          {job.log_path && (
            <div className="card">
              <div className="card__head"><h4>任务产物</h4></div>
              <div className="kv">
                <span className="kv__k">日志</span><span className="kv__v">{job.log_path}</span>
                <span className="kv__k">输出目录</span><span className="kv__v">{job.output_dir}</span>
              </div>
            </div>
          )}
        </div>

        <div className="col">
          <div className="card">
            <div className="card__head"><h4>运行状态</h4></div>
            <div className="stat-row">
              <div className="stat"><span className="stat__k">状态</span><span className="stat__v">{jobLabel}</span></div>
              <div className="stat"><span className="stat__k">任务</span><span className="stat__v" style={{ fontSize: "var(--text-sm)" }}>{job.task_id ?? "—"}</span></div>
            </div>
            <div className="divider" />
            <p className="card__hint">{status}</p>
          </div>

          <div className="card">
            <div className="card__head"><h4>已审批数据</h4></div>
            {datasets.length === 0 ? (
              <p className="card__hint">暂无已审批数据集。完成逐段审核后点击重载。</p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
                {datasets.map((dataset) => (
                  <div className="spk-row" key={dataset.id}>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div className="spk-name">{dataset.task_id}</div>
                      <div className="spk-meta">已通过 {dataset.approved_count}/{dataset.total_count} 段</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
