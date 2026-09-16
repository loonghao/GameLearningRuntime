import { useEffect, useState, type ReactNode } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { StructuredValue } from "./StructuredOutput";
import {
  Brain,
  Terminal,
  Activity,
  GraduationCap,
  Search,
  X,
} from "lucide-react";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { sourceOf, type Event, type Run } from "@/lib/api";

const lanes = ["Model", "Tools", "Learning", "System"] as const;
type Lane = (typeof lanes)[number];
const icons = {
  Model: Brain,
  Tools: Terminal,
  Learning: GraduationCap,
  System: Activity,
};
export function laneOf(event: Event): Lane {
  if (
    event.kind === "agent.decision" ||
    event.kind.startsWith("agent.decision.")
  )
    return "Model";
  if (/^(agent\.execution|tool\.|process\.|bridge\.log)/.test(event.kind))
    return "Tools";
  if (event.kind.startsWith("learning.")) return "Learning";
  return "System";
}
export function durationOf(event: Event): number | null {
  const duration = event.payload.duration_ms;
  return typeof duration === "number" &&
    Number.isFinite(duration) &&
    duration >= 0
    ? duration
    : null;
}
const labelOf = (event: Event) => {
  for (const field of ["label", "summary", "message", "action"])
    if (typeof event.payload[field] === "string")
      return event.payload[field] as string;
  return event.kind;
};
const elapsed = (ms: number) =>
  `${Math.floor(Math.max(0, ms) / 60000)}:${String(Math.floor(Math.max(0, ms) / 1000) % 60).padStart(2, "0")}`;
function Payload({ label, value }: { label: string; value: unknown }) {
  return (
    <section className="trace-payload">
      <h4>{label}</h4>
      <div aria-label={label}>
        <StructuredValue value={value} />
      </div>
    </section>
  );
}
export function ProcessTrace({
  events,
  run,
  selectedEvent,
  onInspect,
  logOutput,
}: {
  events: Event[];
  run: Run | null;
  selectedEvent: Event | null;
  onInspect: (event: Event) => void;
  logOutput?: ReactNode;
}) {
  const [search, setSearch] = useState(""),
    [lane, setLane] = useState("All"),
    [source, setSource] = useState("All"),
    [visible, setVisible] = useState(100),
    [closed, setClosed] = useState(false);
  const [view, setView] = useState("");
  useEffect(() => setVisible(100), [search, lane, source, run?.run_id]);
  useEffect(() => {
    setClosed(false);
    if (selectedEvent) setView("events");
  }, [selectedEvent?.sequence_id, run?.run_id]);
  const sources = [...new Set(events.map(sourceOf))];
  const filtered = events.filter(
    (e) =>
      (lane === "All" || laneOf(e) === lane) &&
      (source === "All" || sourceOf(e) === source) &&
      `${e.kind} ${sourceOf(e)} ${JSON.stringify(e.payload)}`
        .toLowerCase()
        .includes(search.toLowerCase()),
  );
  const rows = filtered.slice(-visible).reverse();
  const start = (run?.started_at_ns ?? events[0]?.timestamp_ns ?? 0) / 1e6;
  const end = Math.max(
    start + 1000,
    ...events.map((e) => e.timestamp_ns / 1e6 + (durationOf(e) ?? 0)),
  );
  const span = end - start;
  const selected = closed ? null : selectedEvent;
  return (
    <section className="process-trace" aria-label="Agent process trace">
      <div className="trace-heading">
        <div>
          <span className="eyebrow">PROCESS</span>
          <h2>Agent activity</h2>
        </div>
        <Badge variant="outline">
          {events.length.toLocaleString()} recorded events
        </Badge>
      </div>
      <Tabs
        value={
          view ||
          (logOutput &&
          !events.some((e) => /^(agent\.|learning\.|tool\.)/.test(e.kind))
            ? "logs"
            : "events")
        }
        onValueChange={setView}
      >
        <TabsList className="activity-sources">
          <TabsTrigger value="events">
            Persisted events ({events.length})
          </TabsTrigger>
          {logOutput && <TabsTrigger value="logs">Process logs</TabsTrigger>}
        </TabsList>
        <TabsContent value="events">
          <div className="trace-overview">
            <div className="trace-ruler">
              <span>Received time</span>
              <div>
                {[0, 0.25, 0.5, 0.75, 1].map((v) => (
                  <small key={v}>{elapsed(v * span)}</small>
                ))}
              </div>
            </div>
            {lanes.map((name) => {
              const items = filtered
                .filter((e) => laneOf(e) === name)
                .slice(-400);
              const Icon = icons[name];
              return (
                <div
                  key={name}
                  className={`trace-lane trace-${name.toLowerCase()}`}
                >
                  <span>
                    <Icon size={13} />
                    {name}
                  </span>
                  <div className="trace-track">
                    {items.map((e) => (
                      <button
                        key={e.sequence_id}
                        className={`trace-span ${selectedEvent?.sequence_id === e.sequence_id ? "selected" : ""}`}
                        style={{
                          left: `${Math.max(0, Math.min(100, ((e.timestamp_ns / 1e6 - start) / span) * 100))}%`,
                          width: `${Math.min(100, ((durationOf(e) ?? 0) / span) * 100)}%`,
                        }}
                        title={`${labelOf(e)} · ${durationOf(e) === null ? "duration not reported" : `${durationOf(e)} ms`}`}
                        aria-label={`Trace ${name} event ${e.sequence_id}`}
                        onClick={() => {
                          setClosed(false);
                          onInspect(e);
                        }}
                      />
                    ))}
                  </div>
                </div>
              );
            })}
            <p className="trace-note">
              Receipt-time markers · bars use reported duration only · latest
              400 matching events per lane.
            </p>
          </div>
          <div className="trace-toolbar">
            <div className="trace-search">
              <Search size={15} />
              <Input
                aria-label="Search process"
                placeholder="Search actions, decisions, output…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
            <select
              aria-label="Process category"
              value={lane}
              onChange={(e) => setLane(e.target.value)}
            >
              {["All", ...lanes].map((l) => (
                <option key={l} value={l}>
                  {l === "All" ? "All categories" : l}
                </option>
              ))}
            </select>
            <select
              aria-label="Process source"
              value={source}
              onChange={(e) => setSource(e.target.value)}
            >
              <option value="All">All sources</option>
              {sources.map((s) => (
                <option key={s}>{s}</option>
              ))}
            </select>
            <span className="muted">{filtered.length} matches</span>
          </div>
          <div className={`trace-body ${selected ? "with-detail" : ""}`}>
            <div className="trace-rows" aria-label="Process steps">
              {rows.map((e) => {
                const Icon = icons[laneOf(e)];
                return (
                  <button
                    key={e.sequence_id}
                    className={`trace-row ${selected?.sequence_id === e.sequence_id ? "selected" : ""}`}
                    aria-label={`Open process event ${e.sequence_id}`}
                    aria-pressed={selected?.sequence_id === e.sequence_id}
                    onClick={() => {
                      setClosed(false);
                      onInspect(e);
                    }}
                  >
                    <time>+{elapsed(e.timestamp_ns / 1e6 - start)}</time>
                    <Icon size={15} />
                    <span className="trace-row-main">
                      <strong>{e.kind}</strong>
                      <span>{labelOf(e)}</span>
                      <small>
                        {sourceOf(e)}
                        {e.episode_id ? ` · ${e.episode_id}` : ""}
                        {e.step_id != null ? ` · step ${e.step_id}` : ""}
                      </small>
                    </span>
                    <span className="trace-duration">
                      {durationOf(e) === null
                        ? "—"
                        : `${(durationOf(e)! / 1000).toFixed(1)}s`}
                    </span>
                  </button>
                );
              })}
              {!rows.length && (
                <p className="trace-empty">
                  {events.length
                    ? "No matching recorded steps."
                    : "Waiting for recorded agent decisions, execution receipts and learning updates."}
                </p>
              )}
              {filtered.length > visible && (
                <Button
                  variant="ghost"
                  onClick={() => setVisible((n) => n + 100)}
                >
                  Load earlier steps
                </Button>
              )}
            </div>
            {selected && (
              <aside
                className="trace-detail"
                aria-label="Process event details"
              >
                <div className="trace-detail-header">
                  <div>
                    <strong>{selected.kind}</strong>
                    <small>
                      {sourceOf(selected)} · #{selected.sequence_id}
                    </small>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label="Close process detail"
                    onClick={() => setClosed(true)}
                  >
                    <X size={16} />
                  </Button>
                </div>
                <Payload
                  label="Recorded summary"
                  value={
                    selected.payload.summary ??
                    selected.payload.message ??
                    selected.payload.label
                  }
                />
                <Payload label="Input" value={selected.payload.input} />
                <Payload
                  label="Output / receipt"
                  value={selected.payload.output ?? selected.payload.receipt}
                />
                <details className="trace-raw">
                  <summary>Full recorded payload</summary>
                  <pre tabIndex={0}>{JSON.stringify(selected, null, 2)}</pre>
                </details>
              </aside>
            )}
          </div>
        </TabsContent>
        {logOutput && (
          <TabsContent
            value="logs"
            forceMount
            className="data-[state=inactive]:hidden"
          >
            {logOutput}
          </TabsContent>
        )}
      </Tabs>
    </section>
  );
}
