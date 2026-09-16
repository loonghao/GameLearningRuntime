import { useState } from "react";
import { Bot, ArrowUpRight, Target } from "lucide-react";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { sourceOf, type BridgeState, type Event, type Run } from "@/lib/api";
import {
  latestSignal,
  parseWorkbench,
  stages,
  type Cell,
  type Section,
} from "@/lib/workbench";

const display = (value: Cell) => (value === null ? "—" : String(value));
function DataSection({ section }: { section: Section }) {
  return (
    <section className={`domain-section domain-${section.kind}`}>
      <h3>{section.title}</h3>
      {section.kind === "stats" && (
        <dl className="domain-stats">
          {section.fields.map((f, i) => (
            <div key={i}>
              <dt>{f.label}</dt>
              <dd>
                {display(f.value)}
                {f.unit && <small> {f.unit}</small>}
              </dd>
            </div>
          ))}
        </dl>
      )}
      {section.kind === "table" && (
        <div
          className="domain-table-scroll"
          tabIndex={0}
          aria-label={section.title}
        >
          <table>
            <thead>
              <tr>
                {section.columns.map((c, i) => (
                  <th key={i} scope="col">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {section.rows.map((row, i) => (
                <tr key={i}>
                  {row.map((c, j) => (
                    <td key={j}>{display(c)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {!section.rows.length && <p className="muted">No rows reported.</p>}
        </div>
      )}
      {section.kind === "text" && <p className="domain-text">{section.text}</p>}
    </section>
  );
}
export function AgentWorkbench({
  run,
  events,
  bridge,
  onInspect,
  onOperate,
}: {
  run: Run | null;
  events: Event[];
  bridge: BridgeState;
  onInspect: (event: Event) => void;
  onOperate?: () => void;
}) {
  const [source, setSource] = useState("");
  const reports = bridge.states
    .filter((e) => e.kind === "bridge.state" && "workbench" in e.payload)
    .map((event) => ({ event, data: parseWorkbench(event.payload.workbench) }));
  const report =
    reports.find((r) => sourceOf(r.event) === source) ?? reports[0];
  const data = report?.data;
  const objective =
    data?.objective ??
    (typeof run?.metadata.objective === "string"
      ? run.metadata.objective
      : undefined);
  return (
    <section className="agent-workbench" aria-label="Agent workbench">
      <div className="agent-mission">
        <div className="agent-mission-copy">
          <span className="eyebrow">
            <Bot size={14} /> AGENT WORKBENCH
          </span>
          <h2>{data?.title ?? "Follow the agent’s training loop"}</h2>
          <p className="agent-objective">
            <Target size={16} />
            {objective ??
              "No objective recorded. Select a run or launch a goal from Operations."}
          </p>
          <div className="agent-context">
            <Badge variant="outline">{run?.status ?? "No run selected"}</Badge>
            {data?.agent && <span>Agent · {data.agent}</span>}
            {data?.phase && <span>Reported phase · {data.phase}</span>}
          </div>
        </div>
        {onOperate && (
          <Button onClick={onOperate}>
            Configure training <ArrowUpRight size={15} />
          </Button>
        )}
      </div>
      <div className="agent-loop" aria-label="Recorded training loop">
        {stages.map(({ title, prefix }, i) => {
          const event = latestSignal(events, prefix);
          return (
            <button
              key={title}
              className={`agent-stage ${event ? "has-signal" : ""}`}
              disabled={!event}
              onClick={() => event && onInspect(event)}
            >
              <span className="agent-stage-number">0{i + 1}</span>
              <span>
                <strong>{title}</strong>
                <small>
                  {event ? event.kind : "No event in loaded window"}
                </small>
                {event && (
                  <small>
                    {sourceOf(event)} ·{" "}
                    {event.step_id === null
                      ? `#${event.sequence_id}`
                      : `step ${event.step_id}`}
                  </small>
                )}
              </span>
              {event && <ArrowUpRight size={14} />}
            </button>
          );
        })}
      </div>
      <p className="agent-scope">
        Latest recorded signal per stage · stages are not a completion
        checklist.
      </p>
      <div className="domain-heading">
        <div>
          <h3>Environment data</h3>
          <p className="muted">Fields and units supplied by the adapter.</p>
        </div>
        {reports.length > 0 && (
          <label>
            Source{" "}
            <select
              aria-label="Workbench source"
              value={report ? sourceOf(report.event) : ""}
              onChange={(e) => setSource(e.target.value)}
            >
              {reports.map((r) => (
                <option key={sourceOf(r.event)}>{sourceOf(r.event)}</option>
              ))}
            </select>
          </label>
        )}
      </div>
      {data ? (
        <div className="domain-grid">
          {data.sections.map((s) => (
            <DataSection key={s.id} section={s} />
          ))}
          {!data.sections.length && (
            <p className="muted">This source has not declared data panels.</p>
          )}
        </div>
      ) : (
        <div className="domain-empty">
          <strong>
            {report
              ? "This reported view cannot be rendered"
              : "Ready for your environment’s data"}
          </strong>
          <p>
            {report
              ? "Unknown or invalid view contract. The original payload remains available in the inspector."
              : "Adapters can describe combat state, economy, inventory, waves, or any other training data through stats, tables and notes. Existing events, metrics and recordings remain available below."}
          </p>
        </div>
      )}
      {report && (
        <div className="agent-provenance">
          <span>
            Reported{" "}
            {new Date(report.event.timestamp_ns / 1e6).toLocaleString()} ·
            diagnostic data, not a live connection check
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onInspect(report.event)}
          >
            Inspect source report
          </Button>
        </div>
      )}
      {bridge.truncated && (
        <p className="muted">
          Source projection is truncated. Query persisted events for complete
          history.
        </p>
      )}
    </section>
  );
}
