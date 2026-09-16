import { useEffect, useRef, useState } from "react";
import {
  Camera,
  Film,
  FileText,
  FolderOpen,
  Download,
  ExternalLink,
  X,
  Crosshair,
  Play,
  Clock3,
} from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "./ui/button";
import { OutputViewer } from "./StructuredOutput";
import { Badge } from "./ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { get, type Event, type Run } from "@/lib/api";

export interface Artifact {
  path: string;
  role: string;
  kind: string;
  mime: string;
  size_bytes: number;
  recorded_sha256: string;
  available: boolean;
  integrity: string;
}
interface MediaPage {
  items: Artifact[];
  next_after: string | null;
}
export interface Frame {
  episode_id: string;
  step_id: number;
  frame_index: number;
  seconds: number;
}
interface FramePage {
  frames: Frame[];
  truncated: boolean;
}
const bytes = (n: number) =>
  n < 1048576
    ? `${(n / 1024).toFixed(1)} KB`
    : `${(n / 1048576).toFixed(1)} MB`;
const name = (path: string) => path.split("/").at(-1) ?? path;
export const mediaUrl = (run: string, path: string) =>
  `/api/v1/media/file?${new URLSearchParams({ run, path })}`;
export function frameForEvent(
  frames: Frame[],
  event: Event,
): Frame | undefined {
  const matches = frames.filter(
    (f) =>
      f.step_id === event.step_id &&
      (!event.episode_id || f.episode_id === event.episode_id),
  );
  return new Set(matches.map((f) => f.episode_id)).size === 1
    ? matches[0]
    : undefined;
}
function relativeArtifact(
  source: string,
  document: string,
  items: Artifact[],
): Artifact | undefined {
  if (!source || /^[a-z][a-z\d+.-]*:|^\/\//i.test(source)) return;
  const base = new URL(document, "https://artifact.invalid/");
  const resolved = new URL(source, base);
  return items.find(
    (a) =>
      `/${a.path}` === decodeURIComponent(resolved.pathname) && a.available,
  );
}

function DocumentView({
  run,
  artifact,
  items,
}: {
  run: string;
  artifact: Artifact;
  items: Artifact[];
}) {
  const [value, setValue] = useState<{
      text: string;
      truncated: boolean;
    } | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    const abort = new AbortController();
    setValue(null);
    setError("");
    get<{ text: string; truncated: boolean }>(
      "media/document",
      { run, path: artifact.path },
      abort.signal,
    )
      .then((v) => {
        if (!abort.signal.aborted) setValue(v);
      })
      .catch((e) => {
        if (!abort.signal.aborted) setError(String(e));
      });
    return () => abort.abort();
  }, [run, artifact.path]);
  return (
    <div className="document-preview">
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {!value && !error && <p className="muted">Loading document…</p>}
      {value &&
        (artifact.kind === "markdown" ? (
          <article className="markdown-content">
            <Markdown
              remarkPlugins={[remarkGfm]}
              skipHtml
              components={{
                img: ({ src, alt }) => {
                  let file: Artifact | undefined;
                  try {
                    file = relativeArtifact(src ?? "", artifact.path, items);
                  } catch {
                    /* Invalid producer links stay inert. */
                  }
                  return file?.kind === "image" ? (
                    <img
                      src={mediaUrl(run, file.path)}
                      alt={alt ?? name(file.path)}
                      loading="lazy"
                    />
                  ) : (
                    <span className="muted">
                      {alt ??
                        "Image unavailable in the loaded artifact library"}
                    </span>
                  );
                },
                a: ({ href, children }) => {
                  let file: Artifact | undefined;
                  try {
                    file = relativeArtifact(href ?? "", artifact.path, items);
                  } catch {
                    /* Invalid producer links stay inert. */
                  }
                  return file ? (
                    <a
                      href={mediaUrl(run, file.path)}
                      download={name(file.path)}
                    >
                      {children}
                    </a>
                  ) : (
                    <span>{children}</span>
                  );
                },
              }}
            >
              {value.text}
            </Markdown>
          </article>
        ) : (
          <OutputViewer text={value.text} partialEnd={value.truncated} />
        ))}
      {value?.truncated && (
        <p className="muted">
          Preview shows the first 256 KiB. Download the file for the complete
          document.
        </p>
      )}
    </div>
  );
}

export function MediaWorkspace({
  run,
  events,
  selectedEvent,
  onInspect,
  onFrame,
}: {
  run: Run | null;
  events: Event[];
  selectedEvent: Event | null;
  onInspect: (event: Event) => void;
  onFrame: (frame: Frame) => void;
}) {
  const [items, setItems] = useState<Artifact[]>([]),
    [after, setAfter] = useState<string | null>(null),
    [error, setError] = useState("");
  const [tab, setTab] = useState("replay"),
    [selected, setSelected] = useState(""),
    [document, setDocument] = useState(""),
    [picture, setPicture] = useState("");
  const [local, setLocal] = useState<{
      url: string;
      name: string;
      kind: string;
    } | null>(null),
    [localError, setLocalError] = useState("");
  const [manifest, setManifest] = useState(""),
    [mapping, setMapping] = useState<FramePage>({
      frames: [],
      truncated: false,
    }),
    [mappingError, setMappingError] = useState(""),
    [time, setTime] = useState(0),
    [playError, setPlayError] = useState("");
  const video = useRef<HTMLVideoElement>(null),
    olderLoaded = useRef(false),
    initialTab = useRef(false);
  useEffect(() => {
    if (initialTab.current || !items.length) return;
    initialTab.current = true;
    setTab(
      items.some((a) => a.kind === "video")
        ? "replay"
        : items.some((a) => a.kind === "image")
          ? "images"
          : "documents",
    );
  }, [items]);
  useEffect(() => {
    if (!run) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const data = await get<MediaPage>(
          "media",
          { run: run!.run_id },
          abort.signal,
        );
        if (abort.signal.aborted) return;
        setItems((old) => [
          ...new Map(
            [...(olderLoaded.current ? old : []), ...data.items].map((a) => [
              a.path,
              a,
            ]),
          ).values(),
        ]);
        if (!olderLoaded.current) setAfter(data.next_after);
        setError("");
      } catch (e) {
        if (!abort.signal.aborted) setError(String(e));
      } finally {
        if (!abort.signal.aborted) timer = setTimeout(poll, 3000);
      }
    }
    void poll();
    return () => {
      abort.abort();
      clearTimeout(timer);
    };
  }, [run?.run_id]);
  useEffect(
    () => () => {
      if (local) URL.revokeObjectURL(local.url);
    },
    [local],
  );
  const videos = items.filter((a) => a.kind === "video"),
    images = items.filter((a) => a.kind === "image"),
    documents = items.filter((a) =>
      ["markdown", "text", "download"].includes(a.kind),
    );
  const active = videos.find((a) => a.path === selected) ?? videos[0],
    activeImage = images.find((a) => a.path === picture) ?? images[0],
    activeDocument = documents.find((a) => a.path === document) ?? documents[0];
  const manifests = items.filter(
    (a) => a.role === "capture-manifest" && a.available,
  );
  const manifestPath =
    manifests.find((a) => a.path === manifest)?.path ??
    manifests.find(
      (a) =>
        a.path.slice(0, a.path.lastIndexOf("/") + 1) ===
        active?.path.slice(0, active.path.lastIndexOf("/") + 1),
    )?.path ??
    "";
  const src =
    local?.kind === "video"
      ? local.url
      : active?.available && run
        ? mediaUrl(run.run_id, active.path)
        : undefined;
  useEffect(() => {
    setMapping({ frames: [], truncated: false });
    setMappingError("");
    if (!run || !active || !manifestPath || local) return;
    const abort = new AbortController();
    get<FramePage>(
      "media/frames",
      { run: run.run_id, manifest: manifestPath, video: active.path },
      abort.signal,
    )
      .then((v) => {
        if (!abort.signal.aborted) setMapping(v);
      })
      .catch((e) => {
        if (!abort.signal.aborted) setMappingError(String(e));
      });
    return () => abort.abort();
  }, [run?.run_id, active?.path, manifestPath, local]);
  useEffect(() => {
    setTime(0);
    setPlayError("");
  }, [src]);
  useEffect(() => {
    if (!selectedEvent || !video.current || local) return;
    const frame = frameForEvent(mapping.frames, selectedEvent);
    if (frame) video.current.currentTime = frame.seconds;
  }, [selectedEvent, mapping.frames, src, local, tab]);
  const currentFrame = mapping.frames.reduce<Frame | undefined>(
    (last, frame) => (frame.seconds <= time ? frame : last),
    undefined,
  );
  const highlights = events
    .filter(
      (e) =>
        e.kind.startsWith("agent.") ||
        e.kind.startsWith("learning.") ||
        e.kind.startsWith("capture."),
    )
    .slice(-8)
    .reverse();
  async function older() {
    if (!run || !after) return;
    try {
      const page = await get<MediaPage>("media", { run: run.run_id, after });
      olderLoaded.current = true;
      setItems((old) => [
        ...new Map([...old, ...page.items].map((a) => [a.path, a])).values(),
      ]);
      setAfter(page.next_after);
    } catch (e) {
      setError(String(e));
    }
  }
  function chooseLocal(file: File | undefined) {
    if (!file) return;
    initialTab.current = true;
    const suffix = file.name.split(".").at(-1)?.toLowerCase();
    const kind = ["mp4", "webm"].includes(suffix ?? "")
      ? "video"
      : ["png", "jpg", "jpeg", "gif", "webp"].includes(suffix ?? "")
        ? "image"
        : null;
    if (!kind) {
      setLocalError("Choose MP4, WebM, PNG, JPEG, WebP or GIF.");
      return;
    }
    setLocal({ url: URL.createObjectURL(file), name: file.name, kind });
    setLocalError("");
    setTab(kind === "video" ? "replay" : "images");
  }
  return (
    <section className="media-workspace" aria-label="Run media and evidence">
      <div className="media-heading">
        <div>
          <div className="eyebrow">RECORD / REPLAY / UNDERSTAND</div>
          <h2>Run replay & evidence</h2>
          <p className="muted">
            The recording, the decision, and the outcome — in one place.
          </p>
        </div>
        <label className="local-picker">
          <FolderOpen size={15} /> Preview local media
          <input
            type="file"
            accept=".mp4,.webm,.png,.jpg,.jpeg,.webp,.gif"
            aria-label="Preview local media"
            onChange={(e) => {
              chooseLocal(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </label>
      </div>
      {local && (
        <div className="local-notice">
          <Badge variant="outline">LOCAL PREVIEW</Badge>
          <span>
            {local.name} · temporary, not saved or associated with this run
          </span>
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Close local preview"
            onClick={() => setLocal(null)}
          >
            <X />
          </Button>
        </div>
      )}
      {(error || localError) && (
        <p role="alert" className="error">
          {error || localError}
        </p>
      )}
      <Tabs
        value={tab}
        onValueChange={(value) => {
          initialTab.current = true;
          setTab(value);
        }}
      >
        <TabsList className="media-tabs">
          <TabsTrigger value="replay">
            <Film size={15} /> Replay <span>{videos.length}</span>
          </TabsTrigger>
          <TabsTrigger value="images">
            <Camera size={15} /> Frames & images <span>{images.length}</span>
          </TabsTrigger>
          <TabsTrigger value="documents">
            <FileText size={15} /> Documents <span>{documents.length}</span>
          </TabsTrigger>
        </TabsList>
        <TabsContent value="replay">
          <div className="replay-layout">
            <div className="replay-main">
              <div className="viewer-toolbar">
                <span className="viewer-label">
                  <Film size={14} />{" "}
                  {local?.kind === "video"
                    ? local.name
                    : active
                      ? name(active.path)
                      : "No registered recording"}
                </span>
                {active && run && !local && (
                  <a
                    href={mediaUrl(run.run_id, active.path)}
                    download={name(active.path)}
                    aria-label="Download recording"
                  >
                    <Download size={15} />
                  </a>
                )}
              </div>
              <div className="video-stage">
                {src ? (
                  <video
                    ref={video}
                    key={src}
                    src={src}
                    controls
                    preload="metadata"
                    playsInline
                    aria-label="Run recording"
                    onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
                    onError={() =>
                      setPlayError(
                        "This recording could not be played. Download it to inspect its format or choose another recording.",
                      )
                    }
                  />
                ) : (
                  <div className="media-empty">
                    <Film size={36} />
                    <h3>A window into this run</h3>
                    <p>
                      Completed recordings registered to the run appear here.
                      Preview an existing local recording with the button above.
                    </p>
                    <span>MP4 / WebM · native playback controls</span>
                  </div>
                )}
              </div>
              {playError && (
                <p role="alert" className="error p-3">
                  {playError}
                </p>
              )}
              <div className="playback-caption">
                <span>
                  <Clock3 size={13} />
                  {time.toFixed(1)}s
                </span>
                <span>
                  {mapping.frames.length
                    ? `${mapping.frames.length.toLocaleString()} mapped frames${mapping.truncated ? " · first 25,000 only" : ""}`
                    : "Review media · no step alignment"}
                </span>
                {currentFrame && !local && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => onFrame(currentFrame)}
                  >
                    <Crosshair />
                    Inspect step {currentFrame.step_id}
                  </Button>
                )}
              </div>
              {videos.length > 0 && (
                <div className="recording-strip">
                  {videos.map((a) => (
                    <button
                      key={a.path}
                      disabled={!a.available}
                      className={a.path === active?.path ? "active" : ""}
                      onClick={() => {
                        setSelected(a.path);
                        setLocal(null);
                      }}
                    >
                      <Play size={13} />
                      <span>
                        {name(a.path)}
                        <small>
                          {bytes(a.size_bytes)} · {a.role}
                        </small>
                      </span>
                    </button>
                  ))}
                </div>
              )}
              {manifests.length > 1 && (
                <label className="field p-3">
                  Capture mapping
                  <select
                    value={manifestPath}
                    onChange={(e) => setManifest(e.target.value)}
                  >
                    {manifests.map((a) => (
                      <option key={a.path}>{a.path}</option>
                    ))}
                  </select>
                </label>
              )}
              {mappingError && (
                <p className="muted p-3">
                  Step alignment unavailable: {mappingError}
                </p>
              )}
            </div>
            <aside className="replay-context">
              <div className="section-heading">
                <h3>Decision moments</h3>
                <Badge variant="outline">{highlights.length}</Badge>
              </div>
              <p className="muted">
                Select a recorded moment to inspect its payload. Indexed
                recordings seek to its step.
              </p>
              <div className="moment-list">
                {highlights.map((e) => (
                  <button
                    key={e.sequence_id}
                    onClick={() => onInspect(e)}
                    className={
                      selectedEvent?.sequence_id === e.sequence_id
                        ? "selected"
                        : ""
                    }
                  >
                    <span
                      className={`moment-marker ${e.kind.startsWith("learning.") ? "learning" : ""}`}
                    />
                    <div>
                      <strong>{e.kind.replaceAll(".", " / ")}</strong>
                      <small>
                        Step {e.step_id ?? "—"} ·{" "}
                        {new Date(e.timestamp_ns / 1e6).toLocaleTimeString()}
                      </small>
                    </div>
                    <ExternalLink size={12} />
                  </button>
                ))}
                {!highlights.length && (
                  <div className="quiet-state">
                    <Crosshair size={22} />
                    <p>No decision moments recorded in this window.</p>
                    <small>
                      Learning and decision events will appear as the producer
                      reports them.
                    </small>
                  </div>
                )}
              </div>
            </aside>
          </div>
        </TabsContent>
        <TabsContent value="images">
          <div className="gallery-layout">
            <div className="image-stage">
              {local?.kind === "image" ? (
                <img src={local.url} alt={local.name} />
              ) : activeImage?.available && run ? (
                <img
                  src={mediaUrl(run.run_id, activeImage.path)}
                  alt={name(activeImage.path)}
                />
              ) : (
                <div className="media-empty">
                  <Camera size={36} />
                  <h3>Keep the decisive frame</h3>
                  <p>
                    Registered screenshots and charts appear here at their
                    original aspect ratio.
                  </p>
                </div>
              )}
              <div className="image-caption">
                {local?.kind === "image"
                  ? local.name
                  : (activeImage?.path ?? "No registered images")}
              </div>
            </div>
            <div className="thumbnail-grid">
              {images.map((a) => (
                <button
                  key={a.path}
                  disabled={!a.available}
                  aria-label={`View ${name(a.path)}`}
                  onClick={() => {
                    setPicture(a.path);
                    setLocal(null);
                  }}
                  className={a.path === activeImage?.path ? "active" : ""}
                >
                  {a.available && run && (
                    <img
                      src={mediaUrl(run.run_id, a.path)}
                      alt=""
                      loading="lazy"
                    />
                  )}
                  <span>
                    {name(a.path)}
                    <small>{bytes(a.size_bytes)}</small>
                  </span>
                </button>
              ))}
            </div>
          </div>
        </TabsContent>
        <TabsContent value="documents">
          <div className="documents-layout">
            <nav aria-label="Run documents">
              {documents.map((a) => (
                <button
                  key={a.path}
                  className={a.path === activeDocument?.path ? "selected" : ""}
                  disabled={!a.available}
                  onClick={() => setDocument(a.path)}
                >
                  <FileText size={16} />
                  <span>
                    {name(a.path)}
                    <small>
                      {a.role} · {bytes(a.size_bytes)}
                    </small>
                  </span>
                </button>
              ))}
              {!documents.length && (
                <p className="muted">No registered documents.</p>
              )}
            </nav>
            <div className="document-surface">
              {activeDocument && run ? (
                <>
                  <div className="viewer-toolbar">
                    <span className="viewer-label">{activeDocument.path}</span>
                    <a
                      href={mediaUrl(run.run_id, activeDocument.path)}
                      download={name(activeDocument.path)}
                    >
                      <Download size={14} /> Download
                    </a>
                  </div>
                  {activeDocument.available &&
                  activeDocument.kind !== "download" ? (
                    <DocumentView
                      run={run.run_id}
                      artifact={activeDocument}
                      items={items}
                    />
                  ) : (
                    <div className="media-empty">
                      <FileText size={32} />
                      <p>
                        {activeDocument.available
                          ? "This file is available as a download."
                          : "The registered file is no longer available."}
                      </p>
                    </div>
                  )}
                </>
              ) : (
                <div className="media-empty">
                  <FileText size={36} />
                  <h3>The story behind the numbers</h3>
                  <p>
                    Read Markdown reports, notes, structured results and logs
                    alongside the run.
                  </p>
                </div>
              )}
            </div>
          </div>
        </TabsContent>
      </Tabs>
      <div className="media-footnote">
        <span>
          {items.length} registered artifacts loaded · preview does not reverify
          file digests
        </span>
        {after && (
          <Button size="sm" variant="outline" onClick={() => void older()}>
            Load more artifacts
          </Button>
        )}
      </div>
    </section>
  );
}
