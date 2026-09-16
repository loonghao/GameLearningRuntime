import { useEffect, useState } from "react";
import { Activity, Download, Pause, Play, Radio, Terminal } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TrainingControls } from "@/components/TrainingControls";
import { MediaWorkspace } from "@/components/MediaWorkspace";
import { AgentWorkbench } from "@/components/AgentWorkbench";
import { ProcessTrace } from "@/components/ProcessTrace";
import {
  BridgePanel,
  EventPanel,
  LogPanel,
  MetricPanel,
  RoutePanel,
} from "@/components/ObservationPanels";
import { download, get, type Event } from "@/lib/api";
import { useObservation, useRuns } from "@/lib/use-observation";

export default function App() {
  const [workspaceTab, setWorkspaceTab] = useState("observe");
  const [health, setHealth] = useState<{
      read_only: boolean;
      environment_id: string;
      version: string;
    } | null>(null),
    [healthError, setHealthError] = useState("");
  const [paused, setPaused] = useState(false),
    [runId, setRunId] = useState<string | null>(null),
    [search, setSearch] = useState("");
  const [detail, setDetail] = useState<unknown>(null),
    [selectedEvent, setSelectedEvent] = useState<Event | null>(null),
    [step, setStep] = useState(""),
    [followJob, setFollowJob] = useState<string | null>(null),
    [historyError, setHistoryError] = useState("");
  const history = useRuns(paused),
    view = useObservation(runId, paused);
  useEffect(() => {
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const data = await get<NonNullable<typeof health>>(
          "health",
          {},
          abort.signal,
        );
        if (!abort.signal.aborted) {
          setHealth(data);
          setHealthError("");
        }
      } catch (e) {
        if (!abort.signal.aborted) {
          setHealthError(String(e));
          timer = setTimeout(load, 2000);
        }
      }
    }
    void load();
    return () => {
      abort.abort();
      clearTimeout(timer);
    };
  }, []);
  useEffect(() => {
    const linked = followJob
      ? history.runs.find((r) => r.metadata.dashboard_job_id === followJob)
      : null;
    if (linked) {
      setRunId(linked.run_id);
      setFollowJob(null);
    } else if (!runId && history.runs.length) setRunId(history.runs[0].run_id);
  }, [history.runs, runId, followJob]);
  useEffect(() => {
    setDetail(null);
    setSelectedEvent(null);
    setStep("");
  }, [runId]);
  const inspect = (event: Event) => {
    setDetail(event);
    setSelectedEvent(event);
    if (event.step_id != null) setStep(String(event.step_id));
  };
  const problem = healthError || history.error || view.error || historyError;
  return (
    <div className="workspace">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">
            <Activity size={22} />
          </span>
          <div>
            <strong>GLR</strong>
            <small>AGENT TRAINING WORKBENCH</small>
          </div>
        </div>
        <div className="sidebar-heading">
          <span>RUN HISTORY</span>
          <Badge variant="outline">{history.runs.length}</Badge>
        </div>
        <Input
          aria-label="Search runs"
          placeholder="Find run, status, type…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <nav aria-label="Run history" className="run-list">
          {history.runs
            .filter((r) =>
              `${r.run_id} ${r.kind} ${r.status}`
                .toLowerCase()
                .includes(search.toLowerCase()),
            )
            .map((r) => (
              <button
                className={`run-entry ${r.run_id === runId ? "selected" : ""}`}
                key={r.run_id}
                onClick={() => {
                  setRunId(r.run_id);
                  setFollowJob(null);
                }}
                aria-current={r.run_id === runId ? "true" : undefined}
              >
                <span className="run-id">{r.run_id}</span>
                <span className="run-caption">
                  <i className={`status-dot ${r.status}`} />
                  {r.kind} · {r.status}
                </span>
                <small>
                  {new Date(r.started_at_ns / 1e6).toLocaleString()}
                </small>
              </button>
            ))}
        </nav>
        {!history.runs.length && (
          <p className="muted">
            No runs recorded yet. Start a training preset.
          </p>
        )}
        {history.before && (
          <Button
            variant="outline"
            onClick={() =>
              void history.older().catch((e) => setHistoryError(String(e)))
            }
          >
            Load older runs
          </Button>
        )}
        <div className="sidebar-footer">
          <Radio size={13} />
          <span>Local workspace · v{health?.version ?? "…"}</span>
        </div>
      </aside>
      <main>
        <header className="workspace-header">
          <div>
            <div className="eyebrow">GOAL / DECIDE / EXECUTE / LEARN</div>
            <h1>Agent training workbench</h1>
            <p className="muted">
              {health?.environment_id ?? "Connecting to GLR…"}
            </p>
          </div>
          <div className="flex flex-wrap gap-2 items-center">
            <Badge variant={problem ? "destructive" : "outline"}>
              {problem
                ? "Connection interrupted"
                : paused
                  ? "Feed paused"
                  : "Observing"}
            </Badge>
            <Button variant="outline" onClick={() => setPaused((p) => !p)}>
              {paused ? <Play /> : <Pause />}
              {paused ? "Resume feed" : "Pause feed"}
            </Button>
            <Button
              variant="outline"
              disabled={!view.run}
              onClick={() =>
                download(
                  {
                    schema_version: "glr.observation-export.v1",
                    scope: "loaded_browser_window",
                    partial: true,
                    run: view.run,
                    events: view.events,
                    metrics: view.metrics,
                    bridge: view.bridge,
                    cursor: view.cursor,
                  },
                  `${runId}-window.json`,
                )
              }
            >
              <Download /> Export window
            </Button>
          </div>
        </header>
        {problem && (
          <p role="alert" className="error banner">
            {problem}
          </p>
        )}
        <Tabs
          value={workspaceTab}
          onValueChange={setWorkspaceTab}
          className="workspace-tabs"
        >
          <TabsList>
            <TabsTrigger value="observe">
              <Activity size={15} /> Agent activity
            </TabsTrigger>
            {health && !health.read_only && (
              <TabsTrigger value="training">
                <Terminal size={15} /> Training & operations
              </TabsTrigger>
            )}
          </TabsList>
          <TabsContent
            value="training"
            forceMount
            className="data-[state=inactive]:hidden"
          >
            {health && !health.read_only && (
              <TrainingControls
                onSelectJob={(id) => {
                  setFollowJob(id);
                  setWorkspaceTab("observe");
                }}
              />
            )}
          </TabsContent>
          <TabsContent value="observe">
            <AgentWorkbench
              key={`agent-${runId}`}
              run={view.run}
              events={view.events}
              bridge={view.bridge}
              onInspect={inspect}
              onOperate={
                health && !health.read_only
                  ? () => setWorkspaceTab("training")
                  : undefined
              }
            />
            <ProcessTrace
              key={`process-${runId}`}
              events={view.events}
              run={view.run}
              selectedEvent={selectedEvent}
              onInspect={inspect}
            />
            <section className="run-summary">
              <div>
                <span className="eyebrow">SELECTED RUN</span>
                <h2>{runId ?? "No run selected"}</h2>
              </div>
              <div className="run-counts">
                <Badge variant="outline">{view.run?.status ?? "Waiting"}</Badge>
                <span>
                  <strong>{view.events.length.toLocaleString()}</strong> events
                  loaded
                </span>
                <span>
                  <strong>{view.metrics.length.toLocaleString()}</strong>{" "}
                  metrics loaded
                </span>
              </div>
            </section>
            <div className="run-facts" aria-label="Run context">
              <div>
                <span>Started</span>
                <strong>
                  {view.run
                    ? new Date(view.run.started_at_ns / 1e6).toLocaleString()
                    : "—"}
                </strong>
              </div>
              <div>
                <span>Run duration</span>
                <strong>
                  {view.run
                    ? `${Math.max(0, Math.floor(((view.run.finished_at_ns ?? Date.now() * 1e6) - view.run.started_at_ns) / 1e9 / 60))} min`
                    : "—"}
                </strong>
              </div>
              <div>
                <span>Episodes in window</span>
                <strong>
                  {
                    new Set(
                      view.events.map((e) => e.episode_id).filter(Boolean),
                    ).size
                  }
                </strong>
              </div>
              <div>
                <span>Latest recorded signal</span>
                <strong>
                  {view.events.at(-1)?.kind ?? "Waiting for events"}
                </strong>
              </div>
            </div>
            <MediaWorkspace
              key={runId}
              run={view.run}
              events={view.events}
              selectedEvent={selectedEvent}
              onInspect={inspect}
              onFrame={(frame) => {
                setStep(String(frame.step_id));
                setDetail({ kind: "capture.frame", ...frame });
              }}
            />
            <div className="section-divider">
              <span>Signals & spatial context</span>
              <small>Persisted observations from the selected run</small>
            </div>
            <div className="signal-grid">
              {view.events.some(
                (e) => e.kind === "navigation.route_sample",
              ) && <RoutePanel events={view.events} onInspect={inspect} />}
              <MetricPanel metrics={view.metrics} />
            </div>
            <BridgePanel bridge={view.bridge} onInspect={inspect} />
            <div className="evidence-grid">
              <EventPanel
                events={view.events}
                onInspect={inspect}
                step={step}
                onStep={setStep}
              />
              <Card>
                <CardHeader>
                  <div className="section-heading">
                    <CardTitle>Evidence inspector</CardTitle>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setDetail(view.run)}
                    >
                      Run metadata
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <pre
                    className="inspector"
                    tabIndex={0}
                    aria-label="Evidence payload"
                  >
                    {detail
                      ? JSON.stringify(detail, null, 2)
                      : "Select an event, Bridge card or route sample to inspect its persisted payload."}
                  </pre>
                </CardContent>
              </Card>
            </div>
            <LogPanel runId={runId} paths={view.logs} paused={paused} />
            <footer className="retention">
              The browser keeps 5,000 events and metrics. Persisted history
              remains queryable through GLR.
              {view.dropped > 0
                ? ` ${view.dropped.toLocaleString()} older records are outside this window.`
                : ""}{" "}
              Export includes only the loaded window. Process status does not
              establish learning quality.
            </footer>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
