import { useEffect, useState } from "react";
import { Button } from "./ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "./ui/card";
import { Badge } from "./ui/badge";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Progress } from "./ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import { sourceOf, type BridgeState, type Event, type Metric } from "@/lib/api";

function coords(event: Event): number[] {
  const p = event.payload.position ?? event.payload.xyz;
  return Array.isArray(p) && p.every(Number.isFinite) ? p : [];
}
const group = (e: Event) =>
  [
    sourceOf(e),
    e.payload.world_id ?? "default",
    e.episode_id ?? "default",
    e.payload.route_id ?? "default",
  ].join(" / ");
export function RoutePanel({
  events,
  onInspect,
}: {
  events: Event[];
  onInspect: (e: Event) => void;
}) {
  const [plane, setPlane] = useState("xy"),
    [selectedGroup, setSelectedGroup] = useState(""),
    [sample, setSample] = useState(0);
  const axes = (
    { xy: [0, 1], xz: [0, 2], yz: [1, 2] } as Record<string, number[]>
  )[plane];
  const samples = events.filter(
    (e) =>
      e.kind === "navigation.route_sample" &&
      axes.every((a) => Number.isFinite(coords(e)[a])),
  );
  const groups = [...new Set(samples.map(group))],
    activeGroup = groups.includes(selectedGroup) ? selectedGroup : groups[0];
  const route = samples.filter((e) => group(e) === activeGroup),
    positions = route.map((e) => axes.map((a) => coords(e)[a]));
  const minX = Math.min(...positions.map((p) => p[0])),
    maxX = Math.max(...positions.map((p) => p[0]));
  const minY = Math.min(...positions.map((p) => p[1])),
    maxY = Math.max(...positions.map((p) => p[1]));
  const scale = Math.max(maxX - minX, ((maxY - minY) * 656) / 256, 1e-9);
  const points = positions.map(([x, y]) => [
    32 + ((x - minX) / scale) * 656,
    288 - ((y - minY) / scale) * 656,
  ]);
  const selected = Math.min(sample, Math.max(route.length - 1, 0));
  return (
    <Card>
      <CardHeader>
        <div className="section-heading">
          <div>
            <CardTitle>Agent route</CardTitle>
            <CardDescription>
              Episode-aligned runtime coordinates
            </CardDescription>
          </div>
          <select
            aria-label="Route plane"
            value={plane}
            onChange={(e) => {
              setPlane(e.target.value);
              setSample(0);
            }}
          >
            <option value="xy">X / Y</option>
            <option value="xz">X / Z</option>
            <option value="yz">Y / Z</option>
          </select>
        </div>
      </CardHeader>
      <CardContent>
        <select
          className="w-full"
          aria-label="Route and episode"
          value={activeGroup ?? ""}
          onChange={(e) => {
            setSelectedGroup(e.target.value);
            setSample(0);
          }}
        >
          {groups.length ? (
            groups.map((g) => <option key={g}>{g}</option>)
          ) : (
            <option value="">No route samples</option>
          )}
        </select>
        <svg
          viewBox="0 0 720 320"
          className="plot route-plot"
          role="img"
          aria-label="Recorded agent route"
        >
          <path d="M32 32V288H688" className="plot-axis" />
          {points.length > 0 && (
            <>
              <polyline
                points={points.map((p) => p.join(",")).join(" ")}
                className="route-line"
              />
              {points.map((p, i) =>
                i % Math.max(1, Math.ceil(points.length / 100)) === 0 ? (
                  <circle
                    key={route[i].sequence_id}
                    cx={p[0]}
                    cy={p[1]}
                    r={3}
                    className="route-point"
                    onClick={() => {
                      setSample(i);
                      onInspect(route[i]);
                    }}
                  />
                ) : null,
              )}
              <circle
                cx={points[selected][0]}
                cy={points[selected][1]}
                r={6}
                className="route-selected"
              />
            </>
          )}
        </svg>
        <Label className="field">
          Scrub recorded samples
          <input
            aria-label="Route sample"
            type="range"
            min={0}
            max={Math.max(0, route.length - 1)}
            value={selected}
            disabled={!route.length}
            onChange={(e) => {
              const i = Number(e.target.value);
              setSample(i);
              onInspect(route[i]);
            }}
          />
        </Label>
        <p className="muted mt-2">
          {route.length
            ? `${route.length} samples · step ${route[selected].step_id ?? "—"} · ${coords(route[selected]).join(", ")}`
            : "No coordinates recorded for this plane. GLR does not infer a route."}
        </p>
      </CardContent>
    </Card>
  );
}
export function MetricPanel({ metrics }: { metrics: Metric[] }) {
  const key = (m: Metric) => `${m.name} · ${m.metadata.source ?? "learner"}`;
  const names = [...new Set(metrics.map(key))],
    [name, setName] = useState("");
  const chosen = names.includes(name) ? name : names[0],
    series = metrics.filter(
      (m) => key(m) === chosen && Number.isFinite(m.value),
    );
  const latest = series.at(-1),
    min = Math.min(...series.map((m) => m.value)),
    max = Math.max(...series.map((m) => m.value));
  const points = series
    .map(
      (m, i) =>
        `${28 + (i / Math.max(1, series.length - 1)) * 424},${208 - ((m.value - min) / Math.max(max - min, 1e-9)) * 160}`,
    )
    .join(" ");
  return (
    <Card>
      <CardHeader>
        <CardTitle>Learning & runtime signals</CardTitle>
        <CardDescription>
          Reported metrics; inspect their diagnostic context
        </CardDescription>
      </CardHeader>
      <CardContent>
        <select
          aria-label="Metric"
          className="w-full"
          value={chosen ?? ""}
          onChange={(e) => setName(e.target.value)}
        >
          {names.length ? (
            names.map((n) => <option key={n}>{n}</option>)
          ) : (
            <option value="">No metrics</option>
          )}
        </select>
        <div className="metric-value">
          {latest
            ? latest.value.toLocaleString(undefined, {
                maximumSignificantDigits: 6,
              })
            : "—"}
        </div>
        <svg
          viewBox="0 0 480 240"
          className="plot"
          role="img"
          aria-label="Recorded metric history"
        >
          <path d="M28 24V208H452" className="plot-axis" />
          {series.length > 0 && (
            <>
              <polyline points={points} className="metric-line" />
              <text x="32" y="20" className="plot-label">
                {max.toPrecision(3)}
              </text>
              <text x="32" y="232" className="plot-label">
                {min.toPrecision(3)}
              </text>
            </>
          )}
        </svg>
        <p className="muted">
          {latest
            ? `${series.length} samples in received order · latest step ${latest.step_id ?? "—"}`
            : "Metrics appear when emitted by a learner or Bridge."}
        </p>
      </CardContent>
    </Card>
  );
}
export function BridgePanel({
  bridge,
  onInspect,
}: {
  bridge: BridgeState;
  onInspect: (e: Event) => void;
}) {
  const [source, setSource] = useState("");
  const sources = [...new Set(bridge.states.map(sourceOf))],
    active = sources.includes(source) ? source : "";
  return (
    <Card>
      <CardHeader>
        <div className="section-heading">
          <div>
            <CardTitle>Bridge telemetry</CardTitle>
            <CardDescription>
              Latest persisted reports. Reported time is not a live connection
              check.
            </CardDescription>
          </div>
          <select
            aria-label="Bridge source"
            value={active}
            onChange={(e) => setSource(e.target.value)}
          >
            <option value="">All sources</option>
            {sources.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </div>
      </CardHeader>
      <CardContent>
        <div className="bridge-grid">
          {bridge.states
            .filter((e) => !active || sourceOf(e) === active)
            .map((e) => {
              const summary =
                e.payload.message ??
                e.payload.state ??
                e.payload.label ??
                "Inspect structured state";
              return (
                <button
                  key={`${sourceOf(e)}:${e.kind}`}
                  className="bridge-card"
                  onClick={() => onInspect(e)}
                >
                  <div className="flex justify-between gap-2">
                    <strong>{sourceOf(e)}</strong>
                    <Badge variant="outline">
                      {e.kind.replace("bridge.", "")}
                    </Badge>
                  </div>
                  <span>
                    {typeof summary === "string"
                      ? summary
                      : JSON.stringify(summary)}
                  </span>
                  <small>
                    Reported {new Date(e.timestamp_ns / 1e6).toLocaleString()} ·
                    step {e.step_id ?? "—"}
                  </small>
                  {e.kind === "bridge.progress" &&
                    typeof e.payload.fraction === "number" && (
                      <Progress
                        aria-label={`${sourceOf(e)} progress`}
                        value={e.payload.fraction * 100}
                      />
                    )}
                </button>
              );
            })}
        </div>
        {!bridge.states.length && (
          <p className="empty-state">
            No Bridge status reported. Custom events and logs appear in the
            timeline.
          </p>
        )}
        {bridge.truncated && (
          <p className="muted">
            First 100 states shown. Query event history for the full record.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
export function EventPanel({
  events,
  onInspect,
  step,
  onStep,
}: {
  events: Event[];
  onInspect: (e: Event) => void;
  step: string;
  onStep: (step: string) => void;
}) {
  const [search, setSearch] = useState("");
  const matches = events.filter(
    (e) =>
      (!step || String(e.step_id) === step) &&
      JSON.stringify(e).toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <Card>
      <CardHeader>
        <CardTitle>Decision & execution timeline</CardTitle>
        <div className="flex flex-wrap gap-3">
          <Input
            aria-label="Search events"
            placeholder="Search event, source or payload…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="flex-1"
          />
          <Input
            aria-label="Filter step"
            placeholder="Step"
            type="number"
            min={0}
            value={step}
            onChange={(e) => onStep(e.target.value)}
            className="w-28"
          />
          <Button
            variant="outline"
            onClick={() => {
              onStep("");
              setSearch("");
            }}
          >
            Clear filters
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="event-scroll">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Sequence</TableHead>
                <TableHead>Step</TableHead>
                <TableHead>Event / source</TableHead>
                <TableHead>Recorded</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {matches
                .slice(-250)
                .reverse()
                .map((e) => (
                  <TableRow key={e.sequence_id}>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onInspect(e)}
                        aria-label={`Inspect event ${e.sequence_id}`}
                      >
                        #{e.sequence_id}
                      </Button>
                    </TableCell>
                    <TableCell>{e.step_id ?? "—"}</TableCell>
                    <TableCell>
                      <strong className="event-kind">{e.kind}</strong>
                      <small className="block muted">{sourceOf(e)}</small>
                    </TableCell>
                    <TableCell>
                      {new Date(e.timestamp_ns / 1e6).toLocaleTimeString()}
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </div>
        <p className="muted mt-3">
          {matches.length
            ? `Showing latest ${Math.min(matches.length, 250)} of ${matches.length} matching loaded events.`
            : "No matching events in the loaded window."}
        </p>
      </CardContent>
    </Card>
  );
}
export { LogPanel } from "./LogPanel";
