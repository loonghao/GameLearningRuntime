import { useEffect, useMemo, useRef, useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "./ui/card";
import { Button } from "./ui/button";
import { OutputViewer } from "./StructuredOutput";
import { get, type LogPage } from "@/lib/api";

function LogSession({
  runId,
  path,
  paused,
}: {
  runId: string;
  path: string;
  paused: boolean;
}) {
  const [pages, setPages] = useState<LogPage[]>([]),
    [size, setSize] = useState(0),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const [query, setQuery] = useState<{
    mode: "live" | "before" | "after";
    boundary?: number;
    id: number;
  }>({ mode: "live", id: 0 });
  const cursor = useRef<number | undefined>(undefined);
  const first = pages[0],
    last = pages.at(-1);
  useEffect(() => {
    if (paused) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function read() {
      setBusy(true);
      try {
        const params: Record<string, string | number> = { run: runId, path };
        if (query.mode === "before") params.before = query.boundary!;
        else if (query.mode === "after") params.offset = query.boundary!;
        else if (cursor.current !== undefined) params.offset = cursor.current;
        const page = await get<LogPage>("log", params, abort.signal);
        if (abort.signal.aborted) return;
        if (query.mode === "live") cursor.current = page.next_offset;
        setSize(page.size_bytes);
        setError("");
        setPages((old) => {
          if (page.reset || !old.length) return [page];
          if (!page.text) return old;
          if (query.mode === "before")
            return page.next_offset === old[0].offset
              ? [page, ...old].slice(0, 16)
              : [page];
          return page.offset === old.at(-1)!.next_offset
            ? [...old, page].slice(-16)
            : [page];
        });
      } catch (e) {
        if (!abort.signal.aborted) setError(String(e));
      } finally {
        if (!abort.signal.aborted) {
          setBusy(false);
          if (query.mode === "live") timer = setTimeout(read, 1000);
        }
      }
    }
    void read();
    return () => {
      abort.abort();
      clearTimeout(timer);
    };
  }, [runId, path, paused, query]);
  const text = useMemo(() => pages.map((p) => p.text).join(""), [pages]);
  function navigate(
    mode: "live" | "before" | "after",
    boundary?: number,
    clear = false,
  ) {
    if (clear) {
      setPages([]);
      cursor.current = undefined;
    }
    setQuery((old) => ({ mode, boundary, id: old.id + 1 }));
  }
  return (
    <>
      <div className="log-navigation">
        <Button
          size="sm"
          variant="outline"
          disabled={paused || busy || !first || first.offset === 0}
          onClick={() => navigate("before", first.offset)}
        >
          Earlier output
        </Button>
        <Button
          size="sm"
          variant="outline"
          disabled={paused || busy || !last || last.next_offset >= size}
          onClick={() => last && navigate("after", last.next_offset)}
        >
          Next output
        </Button>
        <Button
          size="sm"
          variant="ghost"
          disabled={paused || busy}
          onClick={() => navigate("after", 0, true)}
        >
          Read from start
        </Button>
        <Button
          size="sm"
          variant={query.mode === "live" ? "secondary" : "outline"}
          disabled={paused || busy}
          onClick={() => navigate("live", undefined, true)}
        >
          Follow latest
        </Button>
        <span className="muted">
          {paused
            ? "Feed paused"
            : query.mode === "live"
              ? "Following output"
              : "Browsing history"}
        </span>
      </div>
      <p className="output-range">
        Bytes {(first?.offset ?? 0).toLocaleString()}–
        {(last?.next_offset ?? 0).toLocaleString()} / {size.toLocaleString()} ·{" "}
        {path}
      </p>
      {first && first.offset > 0 && (
        <p className="output-notice">
          Earlier bytes are not loaded. Use Earlier output or Read from start;
          the original log remains on disk.
        </p>
      )}
      {last && last.next_offset < size && (
        <p className="output-notice">
          Later bytes are not loaded. Use Next output or Follow latest.
        </p>
      )}
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      <OutputViewer
        key={`${path}:${first?.offset ?? 0}`}
        text={text}
        partialStart={first?.partial_start ?? (first?.offset ?? 0) > 0}
        partialEnd={!!last?.partial_end && last.next_offset < size}
        empty={
          busy ? "Loading recorded output…" : "Waiting for recorded output…"
        }
      />
    </>
  );
}
export function LogPanel({
  runId,
  paths,
  paused,
}: {
  runId: string | null;
  paths: string[];
  paused: boolean;
}) {
  const [selected, setSelected] = useState("");
  const path = paths.includes(selected) ? selected : paths[0];
  return (
    <Card className="managed-log-panel">
      <CardHeader>
        <div className="section-heading">
          <div>
            <CardTitle>Process / FFmpeg output</CardTitle>
            <CardDescription>
              JSON, JSONL, FFmpeg progress and plain output. File records remain
              separate from persisted agent events.
            </CardDescription>
          </div>
          <select
            aria-label="Log file"
            value={path ?? ""}
            onChange={(e) => setSelected(e.target.value)}
          >
            {paths.length ? (
              paths.map((p) => <option key={p}>{p}</option>)
            ) : (
              <option value="">No log files</option>
            )}
          </select>
        </div>
      </CardHeader>
      <CardContent>
        {runId && path ? (
          <LogSession
            key={`${runId}:${path}`}
            runId={runId}
            path={path}
            paused={paused}
          />
        ) : (
          <p className="muted">No managed output for this run.</p>
        )}
      </CardContent>
    </Card>
  );
}
