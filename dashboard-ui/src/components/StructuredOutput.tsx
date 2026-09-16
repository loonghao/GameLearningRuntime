import { useMemo, useState } from "react";
import { Braces, FileText, Search } from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";

const isObject = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object" && !Array.isArray(v);
const scalar = (v: unknown) => (v === null ? "null" : String(v));
function Branch({ label, value }: { label: string; value: unknown }) {
  const [open, setOpen] = useState(false);
  const count = Array.isArray(value)
    ? value.length
    : Object.keys(value as object).length;
  return (
    <details
      className="value-branch"
      onToggle={(e) => setOpen(e.currentTarget.open)}
    >
      <summary>
        {label}{" "}
        <span>
          {count} {Array.isArray(value) ? "items" : "fields"}
        </span>
      </summary>
      {open && <StructuredValue value={value} />}
    </details>
  );
}
export function StructuredValue({ value }: { value: unknown }) {
  const [count, setCount] = useState(30);
  if (value === undefined) return <span className="muted">Not reported</span>;
  if (typeof value !== "object" || value === null)
    return (
      <span className={`value-scalar value-${typeof value}`}>
        {scalar(value)}
      </span>
    );
  if (Array.isArray(value)) {
    if (!value.length) return <span className="muted">Empty list</span>;
    const columns = [
      ...new Set(
        value.slice(0, 200).flatMap((v) => (isObject(v) ? Object.keys(v) : [])),
      ),
    ];
    // Tables are a convenience only when they can preserve every field, including later rows.
    const table =
      columns.length > 0 &&
      columns.length <= 12 &&
      value.every(
        (v) => isObject(v) && Object.keys(v).every((k) => columns.includes(k)),
      );
    return (
      <div className="value-array">
        {table ? (
          <div className="value-table-scroll" tabIndex={0}>
            <table>
              <thead>
                <tr>
                  {columns.map((key) => (
                    <th key={key} scope="col">
                      {key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {value.slice(0, count).map((row, i) => (
                  <tr key={i}>
                    {columns.map((key) => (
                      <td key={key}>
                        {typeof row[key] === "object" && row[key] !== null ? (
                          <Branch label={key} value={row[key]} />
                        ) : (
                          <StructuredValue value={row[key]} />
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          value.slice(0, count).map((item, i) => (
            <div key={i} className="value-array-item">
              {typeof item === "object" && item !== null ? (
                <Branch label={`Item ${i + 1}`} value={item} />
              ) : (
                <>
                  <small>{i + 1}</small>
                  <StructuredValue value={item} />
                </>
              )}
            </div>
          ))
        )}
        {value.length > count && (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setCount((n) => n + 30)}
          >
            Show more items ({value.length - count} remaining)
          </Button>
        )}
      </div>
    );
  }
  const entries = Object.entries(value);
  return (
    <div className="value-object">
      {entries.length === 0 && <span className="muted">Empty object</span>}
      {entries.slice(0, count).map(([key, v]) =>
        typeof v === "object" && v !== null ? (
          <Branch key={key} label={key} value={v} />
        ) : (
          <div className="value-field" key={key}>
            <span className="value-key">{key}</span>
            <StructuredValue value={v} />
          </div>
        ),
      )}
      {entries.length > count && (
        <Button
          size="sm"
          variant="ghost"
          onClick={() => setCount((n) => n + 30)}
        >
          Show more fields ({entries.length - count} remaining)
        </Button>
      )}
    </div>
  );
}
export interface OutputRecord {
  label: string;
  value: unknown;
  raw: string;
  structured: boolean;
  fragment: boolean;
}
export function parseOutput(
  text: string,
  partialStart = false,
  partialEnd = false,
): OutputRecord[] {
  const clean = text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");
  const record = (raw: string, fragment = false): OutputRecord => {
    if (!fragment) {
      try {
        const value: unknown = JSON.parse(raw);
        const label = isObject(value)
          ? [value.kind, value.event, value.type, value.level].find(
              (v) => typeof v === "string",
            )
          : undefined;
        return {
          label: typeof label === "string" ? label : "JSON result",
          value,
          raw,
          structured: true,
          fragment: false,
        };
      } catch {
        /* Plain output remains a first-class record. */
      }
      if (/\b(?:frame|fps|out_time|progress)\s*=/.test(raw)) {
        const fields: Record<string, string> = {};
        for (const match of raw.matchAll(
          /\b([a-zA-Z_][\w]*)\s*=\s*(.*?)(?=\s+[a-zA-Z_][\w]*\s*=|$)/g,
        ))
          fields[match[1]] = match[2].trim();
        if (Object.keys(fields).length)
          return {
            label: "FFmpeg progress",
            value: fields,
            raw,
            structured: true,
            fragment: false,
          };
      }
    }
    return {
      label: fragment ? "Partial record" : "Process output",
      value: raw,
      raw,
      structured: false,
      fragment,
    };
  };
  if (!clean.trim()) return [];
  if (!partialStart && !partialEnd) {
    try {
      JSON.parse(clean);
      return [record(clean)];
    } catch {
      /* Parse mixed JSONL/text below. */
    }
  }
  const lines = clean.split(/\r\n|\n|\r/);
  return lines.flatMap((line, i) =>
    line.trim()
      ? [
          record(
            line,
            (i === 0 && partialStart) || (i === lines.length - 1 && partialEnd),
          ),
        ]
      : [],
  );
}
export function OutputViewer({
  text,
  partialStart = false,
  partialEnd = false,
  empty = "No output yet.",
}: {
  text: string;
  partialStart?: boolean;
  partialEnd?: boolean;
  empty?: string;
}) {
  const [mode, setMode] = useState("structured"),
    [search, setSearch] = useState(""),
    [selected, setSelected] = useState<number | null>(null),
    [count, setCount] = useState(100);
  const records = useMemo(
    () => parseOutput(text, partialStart, partialEnd),
    [text, partialStart, partialEnd],
  );
  const matches = records
    .map((r, index) => ({ ...r, index }))
    .filter((r) =>
      `${r.label} ${r.raw}`.toLowerCase().includes(search.toLowerCase()),
    );
  const active = matches.find((r) => r.index === selected) ?? matches.at(-1);
  const shown = matches.slice(-count).reverse();
  return (
    <div className="output-viewer">
      <div className="output-toolbar">
        <div className="output-modes" role="group" aria-label="Output format">
          <Button
            variant={mode === "structured" ? "secondary" : "ghost"}
            size="sm"
            aria-pressed={mode === "structured"}
            onClick={() => setMode("structured")}
          >
            <Braces size={14} />
            Structured
          </Button>
          <Button
            variant={mode === "raw" ? "secondary" : "ghost"}
            size="sm"
            aria-pressed={mode === "raw"}
            onClick={() => setMode("raw")}
          >
            <FileText size={14} />
            Raw text
          </Button>
        </div>
        <Badge variant="outline">
          {records.length} records in loaded window
        </Badge>
        {mode === "structured" && (
          <div className="output-search">
            <Search size={14} />
            <Input
              aria-label="Search output records"
              value={search}
              placeholder="Search this output window…"
              onChange={(e) => {
                setSearch(e.target.value);
                setCount(100);
              }}
            />
          </div>
        )}
      </div>
      {mode === "raw" ? (
        <pre className="output-raw" tabIndex={0}>
          {text || empty}
        </pre>
      ) : (
        <div
          className={`output-layout ${records.length > 1 ? "with-records" : ""}`}
        >
          {records.length > 1 && (
            <div className="output-records">
              {shown.map((r) => (
                <button
                  key={r.index}
                  aria-label={`View output record ${r.index + 1}`}
                  aria-pressed={active?.index === r.index}
                  className={active?.index === r.index ? "selected" : ""}
                  onClick={() => setSelected(r.index)}
                >
                  <small>#{r.index + 1}</small>
                  <span>
                    <strong>{r.label}</strong>
                    <small>
                      {isObject(r.value)
                        ? Object.entries(r.value)
                            .filter(
                              ([, v]) => v === null || typeof v !== "object",
                            )
                            .slice(0, 4)
                            .map(([k, v]) => `${k}: ${scalar(v)}`)
                            .join(" · ")
                        : r.raw}
                    </small>
                  </span>
                </button>
              ))}
              {matches.length > count && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setCount((n) => n + 100)}
                >
                  Show earlier records ({matches.length - count})
                </Button>
              )}
            </div>
          )}
          <div className="output-record-detail">
            {active ? (
              <>
                <div className="output-record-heading">
                  <strong>{active.label}</strong>
                  <small>Record {active.index + 1} in loaded window</small>
                </div>
                {active.fragment && (
                  <p className="output-notice">
                    This record crosses the loaded byte boundary. Load the
                    adjacent page to reconstruct it.
                  </p>
                )}
                <StructuredValue
                  key={`${active.index}:${active.label}`}
                  value={active.value}
                />
              </>
            ) : (
              <p className="muted">
                {records.length ? "No matching output records." : empty}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
