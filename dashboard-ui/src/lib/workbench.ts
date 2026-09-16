import type { Event } from "./api";

export type Cell = string | number | boolean | null;
export type Section = { id: string; title: string } & (
  | { kind: "stats"; fields: { label: string; value: Cell; unit?: string }[] }
  | { kind: "table"; columns: string[]; rows: Cell[][] }
  | { kind: "text"; text: string }
);
export interface Workbench {
  schema_version: "glr.workbench.v1";
  title: string;
  agent?: string;
  objective?: string;
  phase?: string;
  sections: Section[];
}
const object = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object" && !Array.isArray(v);
const text = (v: unknown, max: number): v is string =>
  typeof v === "string" && v.length > 0 && [...v].length <= max;
const keys = (v: Record<string, unknown>, allowed: string[]) =>
  Object.keys(v).every((k) => allowed.includes(k));
const cell = (v: unknown) =>
  v === null ||
  typeof v === "boolean" ||
  (typeof v === "number" && Number.isFinite(v)) ||
  (typeof v === "string" && [...v].length <= 240);

// Historical or future payloads remain inspectable even when a view cannot be rendered.
export function parseWorkbench(value: unknown): Workbench | null {
  if (
    !object(value) ||
    !keys(value, [
      "schema_version",
      "title",
      "agent",
      "objective",
      "phase",
      "sections",
    ]) ||
    value.schema_version !== "glr.workbench.v1" ||
    !text(value.title, 120) ||
    !Array.isArray(value.sections) ||
    value.sections.length > 8
  )
    return null;
  for (const [key, max] of [
    ["agent", 128],
    ["objective", 2048],
    ["phase", 80],
  ] as const)
    if (value[key] !== undefined && !text(value[key], max)) return null;
  const ids = new Set<string>();
  for (const section of value.sections) {
    if (
      !object(section) ||
      !text(section.id, 64) ||
      !/^[a-zA-Z][a-zA-Z0-9_.-]*$/.test(section.id) ||
      ids.has(section.id) ||
      !text(section.title, 120)
    )
      return null;
    ids.add(section.id);
    if (section.kind === "stats") {
      if (
        !keys(section, ["id", "title", "kind", "fields"]) ||
        !Array.isArray(section.fields) ||
        section.fields.length < 1 ||
        section.fields.length > 12 ||
        !section.fields.every(
          (f: unknown) =>
            object(f) &&
            keys(f, ["label", "value", "unit"]) &&
            text(f.label, 80) &&
            cell(f.value) &&
            (f.unit === undefined || text(f.unit, 32)),
        )
      )
        return null;
    } else if (section.kind === "table") {
      if (
        !keys(section, ["id", "title", "kind", "columns", "rows"]) ||
        !Array.isArray(section.columns) ||
        section.columns.length < 1 ||
        section.columns.length > 8 ||
        !section.columns.every((c: unknown) => text(c, 80)) ||
        !Array.isArray(section.rows) ||
        section.rows.length > 40 ||
        !section.rows.every(
          (r: unknown) =>
            Array.isArray(r) &&
            r.length === (section.columns as unknown[]).length &&
            r.every(cell),
        )
      )
        return null;
    } else if (section.kind === "text") {
      if (
        !keys(section, ["id", "title", "kind", "text"]) ||
        !text(section.text, 4000)
      )
        return null;
    } else return null;
  }
  return value as unknown as Workbench;
}

export const stages = [
  { title: "Goal", prefix: "goal." },
  { title: "Decision", prefix: "agent.decision" },
  { title: "Execution", prefix: "agent.execution" },
  { title: "Learning", prefix: "learning." },
] as const;
export function latestSignal(events: Event[], prefix: string) {
  return events
    .filter(
      (e) =>
        e.kind === prefix ||
        e.kind.startsWith(prefix.endsWith(".") ? prefix : `${prefix}.`),
    )
    .reduce<
      Event | undefined
    >((last, event) => (!last || event.sequence_id > last.sequence_id ? event : last), undefined);
}
