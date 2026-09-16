import { useEffect, useMemo, useRef, useState } from "react";
import {
  Play,
  Download,
  Upload,
  Package,
  Archive,
  Terminal,
} from "lucide-react";
import { Button } from "./ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "./ui/card";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Label } from "./ui/label";
import { Badge } from "./ui/badge";
import {
  control,
  download,
  mergeBy,
  operationArgv,
  type Job,
  type Operation,
  type Preset,
} from "@/lib/api";

export function TrainingControls({
  onSelectJob,
}: {
  onSelectJob: (id: string) => void;
}) {
  const [presets, setPresets] = useState<Preset[]>([]),
    [catalog, setCatalog] = useState<Operation[]>([]);
  const [operationName, setOperationName] = useState("train"),
    [fields, setFields] = useState<Record<string, string | boolean>>({});
  const [presetId, setPresetId] = useState("train.custom"),
    [title, setTitle] = useState("Custom training"),
    [description, setDescription] = useState("");
  const [message, setMessage] = useState(""),
    [error, setError] = useState(""),
    [pending, setPending] = useState(false);
  const [jobs, setJobs] = useState<Job[]>([]),
    [selectedJob, setSelectedJob] = useState(""),
    [before, setBefore] = useState<string | null>(null),
    [stream, setStream] = useState("stderr"),
    [output, setOutput] = useState("");
  const loadedOlder = useRef(false),
    submitLock = useRef(false),
    requestId = useRef<{ key: string; id: string } | null>(null);
  const operation = catalog.find((c) => c.path.join(" ") === operationName);
  const argv = useMemo(
    () => (operation ? operationArgv(operation, fields) : []),
    [operation, fields],
  );
  const training =
    operation?.path[0] === "train" ||
    ["goal run", "task run"].includes(operationName);
  useEffect(() => {
    const abort = new AbortController();
    Promise.all([
      control<Preset[]>("presets", undefined, abort.signal),
      control<{ commands: Operation[] }>("catalog", undefined, abort.signal),
    ])
      .then(([p, c]) => {
        setPresets(p);
        setCatalog(c.commands);
      })
      .catch((e) => {
        if (!abort.signal.aborted) setError(String(e));
      });
    return () => abort.abort();
  }, []);
  useEffect(() => {
    setFields(
      Object.fromEntries(
        (operation?.arguments ?? []).map((f) => [
          f.id,
          f.takes_value ? f.defaults.join(f.multiple ? "\n" : "") : false,
        ]),
      ),
    );
  }, [operation]);
  useEffect(() => {
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const page = await control<{ jobs: Job[]; next_before: string | null }>(
          "jobs",
          undefined,
          abort.signal,
        );
        if (abort.signal.aborted) return;
        setJobs((old) =>
          mergeBy(page.jobs, loadedOlder.current ? old : [], (j) => j.id).sort(
            (a, b) =>
              b.created_at_ms - a.created_at_ms || b.id.localeCompare(a.id),
          ),
        );
        if (!loadedOlder.current) setBefore(page.next_before);
        setSelectedJob((old) => old || page.jobs[0]?.id || "");
      } catch (e) {
        if (!abort.signal.aborted) setError(String(e));
      } finally {
        if (!abort.signal.aborted) timer = setTimeout(poll, 1500);
      }
    }
    void poll();
    return () => {
      abort.abort();
      clearTimeout(timer);
    };
  }, []);
  useEffect(() => {
    setOutput("");
    if (!selectedJob) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const log = await control<{ text: string; tail_truncated: boolean }>(
          `job-log?id=${encodeURIComponent(selectedJob)}&stream=${stream}`,
          undefined,
          abort.signal,
        );
        if (!abort.signal.aborted)
          setOutput(
            (log.tail_truncated ? "[Showing latest 64 KiB]\n" : "") + log.text,
          );
      } catch (e) {
        if (!abort.signal.aborted) setOutput(String(e));
      } finally {
        if (!abort.signal.aborted) timer = setTimeout(poll, 1500);
      }
    }
    void poll();
    return () => {
      abort.abort();
      clearTimeout(timer);
    };
  }, [selectedJob, stream]);
  async function launch(payload: unknown) {
    if (submitLock.current) return;
    submitLock.current = true;
    setPending(true);
    setError("");
    setMessage("Starting operation…");
    const key = JSON.stringify(payload);
    if (requestId.current?.key !== key)
      requestId.current = { key, id: `request-${crypto.randomUUID()}` };
    try {
      const job = await control<Job>("jobs", {
        ...(payload as object),
        request_id: requestId.current.id,
      });
      setJobs((old) => [job, ...old.filter((j) => j.id !== job.id)]);
      setSelectedJob(job.id);
      onSelectJob(job.id);
      setMessage(`${job.id} · ${job.status}`);
      requestId.current = null;
    } catch (e) {
      setError(String(e));
      setMessage("Retrying the same operation will reuse its request ID.");
    } finally {
      submitLock.current = false;
      setPending(false);
    }
  }
  async function save(preset: Preset) {
    setError("");
    try {
      await control("presets", preset);
      setPresets(await control<Preset[]>("presets"));
      setMessage("Preset saved. Review its configuration before starting.");
    } catch (e) {
      setError(String(e));
    }
  }
  async function older() {
    try {
      const page = await control<{ jobs: Job[]; next_before: string | null }>(
        `jobs?before=${encodeURIComponent(before!)}`,
      );
      loadedOlder.current = true;
      setJobs((old) =>
        mergeBy(page.jobs, old, (j) => j.id).sort(
          (a, b) => b.created_at_ms - a.created_at_ms,
        ),
      );
      setBefore(page.next_before);
    } catch (e) {
      setError(String(e));
    }
  }
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="section-heading">
            <div>
              <CardTitle>Training presets</CardTitle>
              <CardDescription>
                Run the project's learner and capture policy.
              </CardDescription>
            </div>
            <Label className="import-button">
              <Upload size={14} /> Import preset
              <Input
                aria-label="Import preset"
                type="file"
                accept="application/json,.json"
                className="sr-only"
                onChange={async (e) => {
                  const file = e.target.files?.[0];
                  e.target.value = "";
                  if (!file) return;
                  try {
                    if (file.size > 65536)
                      throw new Error("Preset exceeds 64 KiB");
                    await save(JSON.parse(await file.text()) as Preset);
                  } catch (err) {
                    setError(String(err));
                  }
                }}
              />
            </Label>
          </div>
        </CardHeader>
        <CardContent>
          <div className="preset-grid">
            {presets.map((p) => (
              <div key={p.id} className="preset-item">
                <Badge variant="outline">{p.id}</Badge>
                <h3>{p.title}</h3>
                <p className="muted">{p.description}</p>
                <div className="flex gap-2">
                  <Button
                    disabled={pending}
                    onClick={() => void launch({ preset: p.id })}
                  >
                    <Play /> Start training
                  </Button>
                  <Button
                    aria-label={`Export ${p.title}`}
                    variant="outline"
                    size="icon"
                    onClick={() =>
                      download(
                        {
                          ...p,
                          id:
                            p.id === "train.default" ? "train.imported" : p.id,
                        },
                        `${p.id}.json`,
                      )
                    }
                  >
                    <Download />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>GLR operations</CardTitle>
          <CardDescription>
            Use the same commands as an agent. Paths refer to this machine.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2 mb-4">
            {[
              ["package export", "Package project", Package],
              ["package import", "Import project", Upload],
              ["backup create", "Back up history", Archive],
              ["backup restore", "Restore history", Archive],
              ["report build", "Build report", Download],
            ].map(([name, label, Icon]) => {
              const Symbol = Icon as typeof Package;
              return (
                <Button
                  key={String(name)}
                  variant="outline"
                  size="sm"
                  onClick={() => setOperationName(String(name))}
                >
                  <Symbol />
                  {String(label)}
                </Button>
              );
            })}
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void launch({ argv });
            }}
            className="space-y-4"
          >
            <Label className="field">
              Operation
              <select
                aria-label="Operation"
                value={operationName}
                onChange={(e) => setOperationName(e.target.value)}
              >
                {catalog.map((c) => (
                  <option key={c.path.join(" ")} value={c.path.join(" ")}>
                    {c.path.join(" › ")}
                  </option>
                ))}
              </select>
            </Label>
            <p className="muted">{operation?.description}</p>
            <div className="form-grid">
              {operation?.arguments.map((f) => (
                <Label key={f.id} className="field" title={f.help}>
                  {f.id.replaceAll("_", " ")}
                  {f.required ? " *" : ""}
                  {!f.takes_value ? (
                    <input
                      type="checkbox"
                      checked={!!fields[f.id]}
                      onChange={(e) =>
                        setFields((old) => ({
                          ...old,
                          [f.id]: e.target.checked,
                        }))
                      }
                    />
                  ) : f.choices.length ? (
                    <select
                      required={f.required}
                      value={String(fields[f.id] ?? "")}
                      onChange={(e) =>
                        setFields((old) => ({ ...old, [f.id]: e.target.value }))
                      }
                    >
                      <option value="">Default</option>
                      {f.choices.map((c) => (
                        <option key={c}>{c}</option>
                      ))}
                    </select>
                  ) : f.multiple ? (
                    <Textarea
                      placeholder="One value per line"
                      required={f.required}
                      value={String(fields[f.id] ?? "")}
                      onChange={(e) =>
                        setFields((old) => ({ ...old, [f.id]: e.target.value }))
                      }
                    />
                  ) : (
                    <Input
                      required={f.required}
                      value={String(fields[f.id] ?? "")}
                      onChange={(e) =>
                        setFields((old) => ({ ...old, [f.id]: e.target.value }))
                      }
                    />
                  )}
                </Label>
              ))}
            </div>
            <pre className="command-preview">
              glr{" "}
              {argv
                .map((a) => (/\s/.test(a) ? JSON.stringify(a) : a))
                .join(" ")}
            </pre>
            <Button disabled={pending || !operation}>
              <Terminal />
              {pending ? "Starting…" : "Run operation"}
            </Button>
          </form>
          <details className="mt-4">
            <summary>Save this training configuration as a preset</summary>
            <div className="form-grid mt-4">
              <Label className="field">
                Preset ID
                <Input
                  value={presetId}
                  onChange={(e) => setPresetId(e.target.value)}
                />
              </Label>
              <Label className="field">
                Title
                <Input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                />
              </Label>
              <Label className="field">
                Description
                <Input
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                />
              </Label>
            </div>
            <Button
              className="mt-3"
              variant="outline"
              disabled={!training || pending}
              onClick={() =>
                void save({
                  schema_version: "glr.training-preset.v1",
                  id: presetId,
                  title,
                  description,
                  argv,
                })
              }
            >
              Save preset
            </Button>
          </details>
        </CardContent>
      </Card>
      {message && (
        <p role="status" className="muted">
          {message}
        </p>
      )}
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Operation history</CardTitle>
          <CardDescription>
            Durable receipts and process output. A completed process does not
            establish learning quality.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="job-layout">
            <div className="job-list">
              {jobs.length === 0 && (
                <p className="muted">No operations submitted.</p>
              )}
              {jobs.map((j) => (
                <Button
                  variant={selectedJob === j.id ? "secondary" : "ghost"}
                  className="job-button"
                  key={j.id}
                  onClick={() => {
                    setSelectedJob(j.id);
                    onSelectJob(j.id);
                  }}
                >
                  <span>{j.requested_argv.join(" ")}</span>
                  <small>
                    {j.observed_status ?? j.status} · exit {j.exit_code ?? "—"}{" "}
                    · {new Date(j.created_at_ms).toLocaleString()}
                  </small>
                </Button>
              ))}
              {before && (
                <Button variant="outline" onClick={() => void older()}>
                  Older operations
                </Button>
              )}
            </div>
            <div className="min-w-0">
              <div className="section-heading">
                <code className="truncate">
                  {selectedJob || "Select an operation"}
                </code>
                <select
                  aria-label="Operation output stream"
                  value={stream}
                  onChange={(e) => setStream(e.target.value)}
                >
                  <option>stderr</option>
                  <option>stdout</option>
                </select>
              </div>
              <pre className="console mt-3" tabIndex={0}>
                {output || "No output yet."}
              </pre>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
