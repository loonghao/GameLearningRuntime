import { useEffect, useRef, useState } from "react";
import {
  get,
  initialCursor,
  mergeBy,
  type BridgeState,
  type Event,
  type Metric,
  type Run,
  type Snapshot,
} from "./api";

export function useRuns(paused: boolean) {
  const [runs, setRuns] = useState<Run[]>([]),
    [before, setBefore] = useState<string | null>(null),
    [error, setError] = useState("");
  const loadedOlder = useRef(false);
  useEffect(() => {
    if (paused) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const page = await get<{ runs: Run[]; next_before: string | null }>(
          "runs",
          {},
          abort.signal,
        );
        if (abort.signal.aborted) return;
        setRuns((old) =>
          mergeBy(
            page.runs,
            loadedOlder.current ? old : [],
            (r) => r.run_id,
          ).sort(
            (a, b) =>
              b.started_at_ns - a.started_at_ns ||
              b.run_id.localeCompare(a.run_id),
          ),
        );
        if (!loadedOlder.current) setBefore(page.next_before);
        setError("");
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
  }, [paused]);
  async function older() {
    if (!before) return;
    const page = await get<{ runs: Run[]; next_before: string | null }>(
      "runs",
      { before },
    );
    loadedOlder.current = true;
    setRuns((old) =>
      mergeBy(page.runs, old, (r) => r.run_id).sort(
        (a, b) =>
          b.started_at_ns - a.started_at_ns || b.run_id.localeCompare(a.run_id),
      ),
    );
    setBefore(page.next_before);
  }
  return { runs, before, older, error };
}
interface View {
  run: Run | null;
  events: Event[];
  metrics: Metric[];
  logs: string[];
  bridge: BridgeState;
  dropped: number;
  error: string;
}
const empty = (): View => ({
  run: null,
  events: [],
  metrics: [],
  logs: [],
  bridge: { states: [], truncated: false },
  dropped: 0,
  error: "",
});
export function useObservation(runId: string | null, paused: boolean) {
  const [view, setView] = useState<View>(empty),
    cursor = useRef(initialCursor());
  useEffect(() => {
    cursor.current = initialCursor();
    setView(empty());
  }, [runId]);
  useEffect(() => {
    if (!runId || paused) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      let delay = 1000;
      try {
        const [page, bridge] = await Promise.all([
          get<Snapshot>(
            "snapshot",
            { run: runId!, ...cursor.current },
            abort.signal,
          ),
          get<BridgeState>("telemetry/state", { run: runId! }, abort.signal),
        ]);
        if (abort.signal.aborted) return;
        cursor.current = page.cursor;
        setView((old) => {
          const events = [...old.events, ...page.events],
            metrics = [...old.metrics, ...page.metrics];
          return {
            run: page.run,
            events: events.slice(-5000),
            metrics: metrics.slice(-5000),
            logs: page.logs,
            bridge,
            dropped:
              old.dropped +
              Math.max(0, events.length - 5000) +
              Math.max(0, metrics.length - 5000),
            error: "",
          };
        });
        if (page.events.length === 250 || page.metrics.length === 250)
          delay = 100;
      } catch (e) {
        if (!abort.signal.aborted)
          setView((old) => ({ ...old, error: String(e) }));
        delay = 2000;
      } finally {
        if (!abort.signal.aborted) timer = setTimeout(poll, delay);
      }
    }
    void poll();
    return () => {
      abort.abort();
      clearTimeout(timer);
    };
  }, [runId, paused]);
  return { ...view, cursor: cursor.current };
}
