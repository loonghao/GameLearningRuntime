import { useCallback, useEffect, useRef, useState } from "react";
import { Clock, RefreshCw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  get,
  type FleetMachine,
  type FleetSnapshot,
  type FleetView,
} from "@/lib/api";

type Age = {
  state: "unknown" | "fresh" | "stale" | "clock_mismatch";
  seconds: number | null;
};
const TTL_SECONDS = 120;
function ageOf(timestamp: string | null, now: number): Age {
  if (
    !timestamp ||
    !Number.isFinite(now) ||
    !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(timestamp)
  )
    return { state: "unknown", seconds: null };
  const parsed = Date.parse(timestamp);
  if (!Number.isFinite(parsed) || new Date(parsed).toISOString() !== timestamp)
    return { state: "unknown", seconds: null };
  if (parsed > now) return { state: "clock_mismatch", seconds: null };
  const seconds = (now - parsed) / 1000;
  return { state: seconds > TTL_SECONDS ? "stale" : "fresh", seconds };
}
function ageLabel(age: Age): string {
  if (age.state === "unknown") return "Unknown";
  if (age.state === "clock_mismatch") return "Clock mismatch";
  const seconds = Math.floor(age.seconds!);
  return seconds < 60 ? seconds + "s ago" : Math.floor(seconds / 60) + "m ago";
}
function TimeEvidence({
  label,
  value,
  now,
}: {
  label: string;
  value: string | null;
  now: number;
}) {
  const age = ageOf(value, now);
  return (
    <div className="fleet-time" aria-label={label}>
      <strong>{ageLabel(age)}</strong>
      <small>{value ?? "Unknown UTC"}</small>
      {age.state === "stale" && <span className="muted">Stale</span>}
    </div>
  );
}
function evidenceState(
  machine: FleetMachine,
  snapshot: Age,
  now: number,
): string {
  if (machine.revoked) return "Revoked";
  if (snapshot.state === "clock_mismatch") return "Clock mismatch";
  const sourceTimes = [
    machine.heartbeat_declared_at_utc,
    machine.heartbeat_received_at_utc,
    machine.data_observed_at_utc,
    machine.data_received_at_utc,
  ];
  if (sourceTimes.some((time) => ageOf(time, now).state === "clock_mismatch"))
    return "Clock mismatch";
  const received = ageOf(machine.heartbeat_received_at_utc, now);
  if (
    received.state === "unknown" ||
    ageOf(machine.heartbeat_declared_at_utc, now).state === "unknown"
  )
    return "Unknown";
  if (snapshot.state !== "fresh")
    return snapshot.state === "stale" ? "Snapshot stale" : "Unknown";
  if (received.state === "stale") return "Heartbeat stale";
  return machine.declared_status === "running"
    ? machine.simulated
      ? "Running (reported)"
      : "Online (reported)"
    : machine.declared_status === "stopped"
      ? "Stopped (reported)"
      : "Unknown";
}
function verdict(machine: FleetMachine, snapshot: Age, now: number): string {
  const evidence = evidenceState(machine, snapshot, now);
  return machine.simulated ? "SIMULATED · " + evidence : evidence;
}
function SourceVersions({ machine }: { machine: FleetMachine }) {
  return (
    <details className="fleet-versions">
      <summary>Versions & source IDs</summary>
      <dl>
        <dt>Source / epoch</dt>
        <dd>
          {machine.source_id} / {machine.source_epoch}
        </dd>
        <dt>Revision</dt>
        <dd>{machine.source_revision}</dd>
        <dt>Runtime commit</dt>
        <dd>{machine.runtime_source_commit}</dd>
        <dt>Adapter SHA256</dt>
        <dd>{machine.adapter_source_sha256}</dd>
        <dt>Behavior policy SHA256</dt>
        <dd>{machine.behavior_policy_sha256}</dd>
        <dt>Checkpoint SHA256</dt>
        <dd>{machine.checkpoint_sha256 ?? "Unknown"}</dd>
        <dt>Heartbeat declared UTC</dt>
        <dd>{machine.heartbeat_declared_at_utc ?? "Unknown"}</dd>
      </dl>
    </details>
  );
}
function SnapshotContent({
  snapshot,
  now,
}: {
  snapshot: FleetSnapshot;
  now: number;
}) {
  const freshness = ageOf(snapshot.generated_at_utc, now);
  return (
    <>
      <div className="fleet-snapshot-status" role="status">
        <Badge
          variant={freshness.state === "fresh" ? "outline" : "destructive"}
        >
          {freshness.state === "fresh"
            ? "Snapshot fresh"
            : freshness.state === "stale"
              ? "Snapshot stale"
              : freshness.state === "clock_mismatch"
                ? "Snapshot clock mismatch"
                : "Snapshot freshness unknown"}
        </Badge>
        <span>
          Generated UTC <code>{snapshot.generated_at_utc}</code> ·{" "}
          {ageLabel(freshness)}
        </span>
        <small>
          Snapshot and heartbeat freshness each expire after 120 seconds.
        </small>
      </div>
      {freshness.state !== "fresh" && (
        <p className="fleet-notice">
          This snapshot cannot establish online machine activity.
        </p>
      )}
      {snapshot.machines.some((machine) => machine.simulated) && (
        <p className="fleet-notice">
          <Badge variant="outline">SIMULATED</Badge> Synthetic sources are
          marked individually and never shown as online.
        </p>
      )}
      <Card className="fleet-section">
        <CardHeader>
          <CardTitle>Machines & runs</CardTitle>
          <p className="muted">
            Heartbeat receipt and data observation are separate evidence.
            Reading this page does not update them.
          </p>
        </CardHeader>
        <CardContent>
          {snapshot.machines.length ? (
            <div className="fleet-table-wrap">
              <table
                className="fleet-table fleet-machines"
                aria-label="Machines and runs"
              >
                <thead>
                  <tr>
                    <th>Machine / run</th>
                    <th>Game / source</th>
                    <th>Evidence state</th>
                    <th>Heartbeat RECEIVED UTC</th>
                    <th>Data OBSERVED UTC</th>
                    <th>Data RECEIVED UTC</th>
                  </tr>
                </thead>
                <tbody>
                  {snapshot.machines.map((machine) => {
                    const status = verdict(machine, freshness, now);
                    return (
                      <tr key={machine.source_id + ":" + machine.source_epoch}>
                        <td data-label="Machine / run">
                          <strong>{machine.machine_id}</strong>
                          <code>{machine.run_id}</code>
                          {machine.simulated && (
                            <Badge variant="outline">SIMULATED</Badge>
                          )}
                        </td>
                        <td data-label="Game / source">
                          <strong>{machine.game_id}</strong>
                          <small>{machine.environment_id}</small>
                          <SourceVersions machine={machine} />
                        </td>
                        <td data-label="Evidence state">
                          <Badge
                            variant={
                              status === "Online (reported)"
                                ? "default"
                                : machine.revoked ||
                                    status.endsWith("Clock mismatch")
                                  ? "destructive"
                                  : "outline"
                            }
                          >
                            {status}
                          </Badge>
                          <small>Declared: {machine.declared_status}</small>
                        </td>
                        <td data-label="Heartbeat RECEIVED UTC">
                          <TimeEvidence
                            label={machine.source_id + " heartbeat received"}
                            value={machine.heartbeat_received_at_utc}
                            now={now}
                          />
                        </td>
                        <td data-label="Data OBSERVED UTC">
                          <TimeEvidence
                            label={machine.source_id + " data observed"}
                            value={machine.data_observed_at_utc}
                            now={now}
                          />
                        </td>
                        <td data-label="Data RECEIVED UTC">
                          <TimeEvidence
                            label={machine.source_id + " data received"}
                            value={machine.data_received_at_utc}
                            now={now}
                          />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="fleet-empty">
              No registered sources in this snapshot. Machine activity is
              unknown.
            </p>
          )}
        </CardContent>
      </Card>
      <Card className="fleet-section">
        <CardHeader>
          <CardTitle>Dataset compatibility & quality</CardTitle>
          <p className="muted">
            Exact compatibility cohorts are separate. Accepted data and
            selection eligibility do not establish learner consumption.
          </p>
        </CardHeader>
        <CardContent>
          {snapshot.datasets.length ? (
            <div className="fleet-table-wrap">
              <table
                className="fleet-table"
                aria-label="Dataset compatibility and quality"
              >
                <thead>
                  <tr>
                    <th>Assignment / source epoch</th>
                    <th>Compatibility group</th>
                    <th>Split / quality</th>
                    <th>Accepted / duplicate</th>
                    <th>Ready / rejected shards</th>
                    <th>Last learner selection</th>
                  </tr>
                </thead>
                <tbody>
                  {snapshot.datasets.map((dataset) => (
                    <tr
                      key={[
                        dataset.source_id,
                        dataset.source_epoch,
                        dataset.compatibility_group_sha256,
                        dataset.assignment_id,
                        dataset.split,
                      ].join(":")}
                    >
                      <td data-label="Assignment / source epoch">
                        <strong>{dataset.assignment_id}</strong>
                        <code>
                          {dataset.source_id} / {dataset.source_epoch}
                        </code>
                      </td>
                      <td data-label="Compatibility group">
                        <code className="fleet-digest">
                          {dataset.compatibility_group_sha256}
                        </code>
                      </td>
                      <td data-label="Split / quality">
                        <Badge
                          variant={
                            dataset.quality_status === "revoked" ||
                            dataset.quality_status === "quarantine"
                              ? "destructive"
                              : "outline"
                          }
                        >
                          {dataset.quality_status}
                        </Badge>
                        <small>
                          {dataset.split === "evaluation_holdout"
                            ? "Evaluation holdout"
                            : dataset.split}
                        </small>
                      </td>
                      <td data-label="Accepted / duplicate">
                        <strong>
                          {dataset.accepted_transition_count.toLocaleString()}{" "}
                          accepted
                        </strong>
                        <small>
                          {dataset.duplicate_transition_count.toLocaleString()}{" "}
                          duplicate
                        </small>
                      </td>
                      <td data-label="Ready / rejected shards">
                        <strong>
                          {dataset.ready_shard_count.toLocaleString()} ready
                        </strong>
                        <small>
                          {dataset.rejected_shard_count.toLocaleString()}{" "}
                          rejected · {dataset.retained_bytes.toLocaleString()}{" "}
                          bytes retained
                        </small>
                      </td>
                      <td data-label="Last learner selection">
                        <strong>
                          {dataset.last_plan_eligible === null
                            ? "Not evaluated"
                            : dataset.last_plan_eligible
                              ? "Eligible"
                              : "Ineligible"}
                        </strong>
                        <small>
                          {dataset.last_plan_reason_codes.length
                            ? dataset.last_plan_reason_codes
                                .map((reason) => reason.replaceAll("_", " "))
                                .join(", ")
                            : "No recorded selection reasons"}
                        </small>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="fleet-empty">
              No dataset quality records in this snapshot.
            </p>
          )}
        </CardContent>
      </Card>
      <Card className="fleet-section">
        <CardHeader>
          <CardTitle>Durable consumer receipts</CardTitle>
          <p className="muted">
            Receipt source labels do not identify the current epoch or run.
            Epoch scope is held in the durable plan and is not included in this
            snapshot.
          </p>
        </CardHeader>
        <CardContent>
          {snapshot.consumer_receipts.length ? (
            <div className="fleet-table-wrap">
              <table
                className="fleet-table"
                aria-label="Durable consumer receipts"
              >
                <thead>
                  <tr>
                    <th>Receipt / plan</th>
                    <th>Learner / source labels</th>
                    <th>Recorded result</th>
                    <th>Callback completion</th>
                    <th>Learner declared updates</th>
                    <th>Finished UTC</th>
                  </tr>
                </thead>
                <tbody>
                  {snapshot.consumer_receipts.map((receipt) => (
                    <tr key={receipt.receipt_id}>
                      <td data-label="Receipt / plan">
                        <strong>{receipt.receipt_id}</strong>
                        <code>{receipt.plan_id}</code>
                      </td>
                      <td data-label="Learner / source labels">
                        <strong>{receipt.learner_id}</strong>
                        <code>
                          {receipt.source_ids.length
                            ? receipt.source_ids.join(", ")
                            : "No source labels recorded"}
                        </code>
                      </td>
                      <td data-label="Recorded result">
                        <Badge
                          variant={
                            receipt.status === "rejected"
                              ? "destructive"
                              : "outline"
                          }
                        >
                          {receipt.status === "consumed"
                            ? "Consumed (recorded)"
                            : receipt.status === "unknown_effect"
                              ? "Effect unknown"
                              : "Rejected"}
                        </Badge>
                        <small>
                          {receipt.transition_count.toLocaleString()}{" "}
                          transitions
                        </small>
                      </td>
                      <td data-label="Callback completion">
                        <strong>
                          {receipt.callback_completed
                            ? "Observed completed"
                            : "Not observed completed"}
                        </strong>
                      </td>
                      <td data-label="Learner declared updates">
                        <strong>
                          {receipt.learner_declared_updates === null
                            ? "Unknown"
                            : receipt.learner_declared_updates.toLocaleString()}
                        </strong>
                        <small>Declaration, not verified improvement</small>
                      </td>
                      <td data-label="Finished UTC">
                        <TimeEvidence
                          label={receipt.receipt_id + " finished"}
                          value={receipt.finished_at_utc}
                          now={now}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="fleet-empty">
              No durable consumer receipts in this snapshot. Consumption is
              unknown.
            </p>
          )}
        </CardContent>
      </Card>
    </>
  );
}
export function FleetWorkspace() {
  const [view, setView] = useState<FleetView | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [now, setNow] = useState(Date.now);
  const active = useRef<AbortController | null>(null);
  const refresh = useCallback(async () => {
    active.current?.abort();
    const abort = new AbortController();
    active.current = abort;
    setLoading(true);
    setFailed(false);
    try {
      const result = await get<FleetView>("fleet", {}, abort.signal);
      if (
        result.schema_version !== "glr.fleet.view.v1" ||
        !["available", "missing", "invalid", "unavailable"].includes(
          result.status,
        ) ||
        (result.status === "available"
          ? result.snapshot?.schema_version !== "glr.fleet.snapshot.v1"
          : result.snapshot !== null)
      )
        throw new Error("Invalid fleet view");
      if (!abort.signal.aborted) {
        setView(result);
        setNow(Date.now());
      }
    } catch {
      if (!abort.signal.aborted) {
        setFailed(true);
        setView(null);
      }
    } finally {
      if (!abort.signal.aborted) setLoading(false);
    }
  }, []);
  useEffect(() => {
    void refresh();
    const clock = setInterval(() => setNow(Date.now()), 10000);
    return () => {
      active.current?.abort();
      clearInterval(clock);
    };
  }, [refresh]);
  useEffect(() => {
    if (view?.status !== "available") return;
    const clock = Date.now();
    const deadlines = [
      view.snapshot.generated_at_utc,
      ...view.snapshot.machines.map(
        (machine) => machine.heartbeat_received_at_utc,
      ),
    ]
      .filter((time) => ageOf(time, clock).state === "fresh")
      .map((time) => Date.parse(time!) + TTL_SECONDS * 1000 + 1 - clock);
    if (!deadlines.length) return;
    const expiry = setTimeout(() => setNow(Date.now()), Math.min(...deadlines));
    return () => clearTimeout(expiry);
  }, [view, now]);
  return (
    <section className="fleet-workspace" aria-label="Fleet workspace">
      <div className="section-heading fleet-heading">
        <div>
          <div className="eyebrow">LOCAL TRUSTED SOURCES</div>
          <h2>Fleet</h2>
          <p className="muted">
            Machine, run and dataset evidence from the local snapshot.
          </p>
        </div>
        <Button
          variant="outline"
          disabled={loading}
          onClick={() => void refresh()}
        >
          <RefreshCw size={15} /> Refresh snapshot
        </Button>
      </div>
      <p className="fleet-read-only">
        <Clock size={14} /> Read only · manual refresh · no remote collection or
        training controls
      </p>
      {loading && <p role="status">Reading fleet snapshot…</p>}
      {failed && (
        <p role="alert" className="error banner">
          Fleet snapshot unavailable. Machine activity is unknown.
        </p>
      )}
      {!failed && view?.status !== "available" && !loading && (
        <p role="status" className="fleet-empty">
          {view?.status === "missing"
            ? "No fleet snapshot recorded. Machine activity is unknown."
            : view?.status === "invalid"
              ? "Fleet snapshot rejected. Machine activity is unknown."
              : "Fleet snapshot unavailable. Machine activity is unknown."}
        </p>
      )}
      {!failed && view?.status === "available" && (
        <SnapshotContent snapshot={view.snapshot} now={now} />
      )}
    </section>
  );
}
