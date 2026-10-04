export interface Run {
  run_id: string;
  environment_id: string;
  kind: string;
  status: string;
  started_at_ns: number;
  finished_at_ns?: number | null;
  metadata: Record<string, unknown>;
}
export interface Event {
  sequence_id: number;
  timestamp_ns: number;
  kind: string;
  step_id: number | null;
  episode_id?: string | null;
  payload: Record<string, unknown>;
  source?: string;
}
export interface Metric {
  metric_id: number;
  timestamp_ns: number;
  name: string;
  value: number;
  step_id: number | null;
  metadata: Record<string, unknown>;
}
export interface Cursor {
  events_after: number;
  metrics_after: number;
}
export interface Snapshot {
  run: Run;
  events: Event[];
  metrics: Metric[];
  logs: string[];
  cursor: Cursor;
}
export interface BridgeState {
  states: Event[];
  truncated: boolean;
}
export interface FleetMachine {
  source_id: string;
  source_epoch: string;
  machine_id: string;
  simulated: boolean;
  revoked: boolean;
  declared_status: "unknown" | "running" | "stopped";
  run_id: string;
  game_id: string;
  environment_id: string;
  source_revision: string;
  runtime_source_commit: string;
  adapter_source_sha256: string;
  behavior_policy_sha256: string;
  checkpoint_sha256: string | null;
  heartbeat_declared_at_utc: string | null;
  heartbeat_received_at_utc: string | null;
  data_observed_at_utc: string | null;
  data_received_at_utc: string | null;
}
export type FleetPlanReason =
  | "eligible"
  | "holdout"
  | "quarantine"
  | "revoked"
  | "compatibility_mismatch"
  | "policy_mismatch"
  | "off_policy_not_allowed"
  | "simulated_not_allowed"
  | "already_claimed"
  | "no_ready_data";
export interface FleetDataset {
  source_id: string;
  source_epoch: string;
  compatibility_group_sha256: string;
  assignment_id: string;
  split: "train" | "evaluation_holdout" | "quarantine";
  quality_status: "empty" | "ready" | "quarantine" | "revoked";
  accepted_transition_count: number;
  duplicate_transition_count: number;
  rejected_shard_count: number;
  ready_shard_count: number;
  retained_bytes: number;
  last_plan_eligible: boolean | null;
  last_plan_reason_codes: FleetPlanReason[];
}
export interface FleetConsumerReceipt {
  receipt_id: string;
  plan_id: string;
  learner_id: string;
  source_ids: string[];
  status: "consumed" | "unknown_effect" | "rejected";
  transition_count: number;
  callback_completed: boolean;
  learner_declared_updates: number | null;
  finished_at_utc: string | null;
}
export interface FleetSnapshot {
  schema_version: "glr.fleet.snapshot.v1";
  generated_at_utc: string;
  scope: "local_trusted_sources";
  heartbeat_ttl_seconds: 120;
  machines: FleetMachine[];
  datasets: FleetDataset[];
  consumer_receipts: FleetConsumerReceipt[];
}
export type FleetView = {
  schema_version: "glr.fleet.view.v1";
} & (
  | { status: "available"; snapshot: FleetSnapshot }
  | { status: "missing" | "invalid" | "unavailable"; snapshot: null }
);
export interface LogPage {
  text: string;
  offset: number;
  partial_start?: boolean;
  partial_end?: boolean;
  tail_truncated?: boolean;
  more?: boolean;
  reset: boolean;
  next_offset: number;
  size_bytes: number;
}
export interface Preset {
  schema_version: string;
  id: string;
  title: string;
  description: string;
  argv: string[];
}
export interface Job {
  id: string;
  status: string;
  observed_status?: string;
  requested_argv: string[];
  created_at_ms: number;
  exit_code: number | null;
}
export interface Argument {
  id: string;
  long: string | null;
  takes_value: boolean;
  required: boolean;
  multiple: boolean;
  repeat: boolean;
  defaults: string[];
  choices: string[];
  help: string;
}
export interface Operation {
  path: string[];
  description: string;
  arguments: Argument[];
}
export const initialCursor = (): Cursor => ({
  events_after: -1,
  metrics_after: 0,
});

async function request<T>(url: string, init: RequestInit = {}): Promise<T> {
  const timeout = AbortSignal.timeout(8000);
  const response = await fetch(url, {
    ...init,
    signal: init.signal ? AbortSignal.any([init.signal, timeout]) : timeout,
  });
  const value = await response.json();
  if (!response.ok)
    throw new Error(
      value.error?.message ?? value.error ?? `HTTP ${response.status}`,
    );
  return value as T;
}
export function get<T>(
  path: string,
  params: Record<string, string | number> = {},
  signal?: AbortSignal,
) {
  return request<T>(
    `/api/v1/${path}?${new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]))}`,
    { signal },
  );
}
export async function control<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const response = await request<{ data: T }>(
    `/api/v1/control/${path}`,
    body === undefined
      ? { signal }
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal,
        },
  );
  return response.data;
}
export function download(value: unknown, name: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function mergeBy<T>(latest: T[], older: T[], id: (v: T) => string): T[] {
  return [...new Map([...older, ...latest].map((v) => [id(v), v])).values()];
}
export function operationArgv(
  operation: Operation,
  fields: Record<string, string | boolean>,
): string[] {
  const argv = [...operation.path];
  for (const field of operation.arguments) {
    const value = fields[field.id];
    if (!field.takes_value) {
      if (value && field.long) argv.push(`--${field.long}`);
      continue;
    }
    if (typeof value !== "string" || !value.trim()) continue;
    const values = field.multiple
      ? value
          .split("\n")
          .map((v) => v.trim())
          .filter(Boolean)
      : [value];
    if (field.repeat && field.long)
      for (const v of values) argv.push(`--${field.long}`, v);
    else {
      if (field.long) argv.push(`--${field.long}`);
      argv.push(...values);
    }
  }
  return argv;
}
export function sourceOf(event: Event): string {
  const metadata = event.payload._glr as { source?: string } | undefined;
  return event.source ?? metadata?.source ?? "learner";
}
