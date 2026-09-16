/* GLR has no browser-side execution authority. All rendered values are text. */
"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  runs: [],
  run: null,
  events: [],
  metrics: [],
  cursor: { events_after: -1, metrics_after: 0 },
  paused: false,
  generation: 0,
  logOffset: null,
  logPath: "",
  dropped: 0,
  followJob: null,
  olderRuns: [],
  nextBefore: null,
};
const text = (id, value) => {
  $(id).textContent = String(value);
};
const date = (ns) => new Date(ns / 1e6).toLocaleTimeString();
async function api(path, params = {}) {
  const response = await fetch(
    `/api/v1/${path}?${new URLSearchParams(params)}`,
    { signal: AbortSignal.timeout(8000) },
  );
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      data.error?.message || data.error || `HTTP ${response.status}`,
    );
  return data;
}
function options(id, values, empty) {
  const element = $(id),
    selected = element.value;
  if (
    JSON.stringify([...element.options].map((o) => o.value)) ===
    JSON.stringify(values)
  )
    return;
  element.replaceChildren();
  for (const value of values.length ? values : [""]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value || empty;
    element.append(option);
  }
  if (values.includes(selected)) element.value = selected;
}
function selectRun(run) {
  state.generation++;
  state.run = run;
  state.events = [];
  state.metrics = [];
  state.cursor = { events_after: -1, metrics_after: 0 };
  state.logOffset = null;
  state.logPath = "";
  state.dropped = 0;
  text("detail", JSON.stringify(run, null, 2));
  text("log-text", "Waiting for output…");
  $("step").value = "";
  options("log", [], "No logs");
  options("metric", [], "No metrics");
  options("route-group", [], "No samples");
  render();
  renderRuns();
}
function renderRuns() {
  const filter = $("run-search").value.toLowerCase();
  $("runs").replaceChildren();
  for (const run of state.runs.filter((r) =>
    `${r.run_id} ${r.status} ${r.kind}`.toLowerCase().includes(filter),
  )) {
    const b = document.createElement("button");
    b.className = `run-button${state.run?.run_id === run.run_id ? " selected" : ""}`;
    b.textContent = run.run_id;
    const meta = document.createElement("small");
    meta.textContent = `${run.kind} · ${run.status} · ${new Date(run.started_at_ns / 1e6).toLocaleString()}`;
    b.append(meta);
    b.onclick = () => selectRun(run);
    $("runs").append(b);
  }
  if (!$("runs").childNodes.length)
    text("runs", "No matching runs. Start training to record a run.");
}
const stepOf = (e) => e.step_id ?? e.payload?.step_id;
function svgNode(tag, attrs, parent) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  parent.append(n);
  return n;
}
function projectPoints(points, width, height) {
  const xs = points.map((p) => p[0]),
    ys = points.map((p) => p[1]);
  const minX = Math.min(...xs),
    maxX = Math.max(...xs),
    minY = Math.min(...ys),
    maxY = Math.max(...ys);
  return {
    points: points.map((p) => [
      32 + ((p[0] - minX) / Math.max(maxX - minX, 1e-9)) * (width - 64),
      height -
        32 -
        ((p[1] - minY) / Math.max(maxY - minY, 1e-9)) * (height - 64),
    ]),
    minX,
    maxX,
    minY,
    maxY,
  };
}
function inspect(event) {
  text("detail", JSON.stringify(event, null, 2));
  if (stepOf(event) != null) $("step").value = stepOf(event);
  renderEvents();
}
function renderRoute() {
  const plane = $("plane").value,
    axes = { xy: [0, 1], xz: [0, 2], yz: [1, 2] }[plane];
  const routes = state.events
    .filter((e) => e.kind === "navigation.route_sample")
    .filter((e) => {
      const p = e.payload?.position ?? e.payload?.xyz;
      return Array.isArray(p) && axes.every((i) => Number.isFinite(p[i]));
    });
  const group = (e) =>
    JSON.stringify([
      e.payload?.world_id ?? "default",
      e.episode_id ?? e.payload?.episode_id ?? "default",
      e.payload?.route_id ?? "default",
    ]);
  options("route-group", [...new Set(routes.map(group))], "No samples");
  const events = routes.filter((e) => group(e) === $("route-group").value),
    svg = $("route");
  svg.replaceChildren();
  $("scrub").disabled = !events.length;
  $("scrub").max = Math.max(0, events.length - 1);
  if (!events.length) {
    text(
      "route-note",
      "No route samples for this plane. GLR does not infer coordinates.",
    );
    return;
  }
  const plotted = projectPoints(
    events.map((e) => {
      const p = e.payload.position ?? e.payload.xyz;
      return axes.map((i) => p[i]);
    }),
    720,
    320,
  );
  // Preserve physical aspect ratio so a straight route is not visually distorted.
  const span = Math.max(
    plotted.maxX - plotted.minX,
    ((plotted.maxY - plotted.minY) * 656) / 256,
    1e-9,
  );
  const points = events.map((e) => {
    const p = e.payload.position ?? e.payload.xyz;
    return [
      32 + ((p[axes[0]] - plotted.minX) / span) * 656,
      288 - ((p[axes[1]] - plotted.minY) / span) * 656,
    ];
  });
  svgNode(
    "polyline",
    { points: points.map((p) => p.join(",")).join(" ") },
    svg,
  );
  // Keep SVG interaction bounded; the scrubber reaches every retained sample.
  const stride = Math.max(1, Math.ceil(points.length / 300));
  points.forEach((p, i) => {
    if (i % stride && i !== points.length - 1) return;
    const circle = svgNode(
      "circle",
      {
        cx: p[0],
        cy: p[1],
        r: i === points.length - 1 ? 6 : 4,
        tabindex: 0,
        role: "button",
        "aria-label": `Route sample, step ${stepOf(events[i]) ?? "unknown"}`,
      },
      svg,
    );
    circle.onclick = () => inspect(events[i]);
    circle.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        inspect(events[i]);
      }
    };
    svgNode("title", {}, circle).textContent =
      `step ${stepOf(events[i]) ?? "—"} · ${JSON.stringify(events[i].payload.position ?? events[i].payload.xyz)}`;
  });
  $("scrub").oninput = () => inspect(events[Number($("scrub").value)]);
  text(
    "route-note",
    `${events.length} recorded samples · ${plane.toUpperCase()} · select a point or scrub to inspect its step`,
  );
}
function renderMetrics() {
  options(
    "metric",
    [...new Set(state.metrics.map((m) => m.name))],
    "No metrics",
  );
  const data = state.metrics.filter((m) => m.name === $("metric").value),
    svg = $("chart");
  svg.replaceChildren();
  if (!data.length) {
    text("metric-value", "—");
    text("metric-note", "No learner metrics recorded.");
    return;
  }
  const last = data.at(-1);
  text("metric-value", Number(last.value).toPrecision(5));
  const points = projectPoints(
    data.map((m) => [m.metric_id, m.value]),
    480,
    240,
  );
  svgNode(
    "polyline",
    { points: points.points.map((p) => p.join(",")).join(" ") },
    svg,
  );
  svgNode("text", { x: 12, y: 20 }, svg).textContent =
    `max ${points.maxY.toPrecision(4)}`;
  svgNode("text", { x: 12, y: 230 }, svg).textContent =
    `min ${points.minY.toPrecision(4)}`;
  text(
    "metric-note",
    `${data.length} samples · metric ID ${last.metric_id} · step ${last.step_id ?? "—"} · ${date(last.timestamp_ns)}`,
  );
}
function renderEvents() {
  const search = $("filter").value.toLowerCase(),
    step = $("step").value;
  const events = state.events
    .filter(
      (e) =>
        (!step || String(stepOf(e)) === step) &&
        (!search || JSON.stringify(e).toLowerCase().includes(search)),
    )
    .slice(-250)
    .reverse();
  $("events").replaceChildren();
  for (const event of events) {
    const row = document.createElement("tr");
    for (const value of [
      event.sequence_id,
      stepOf(event) ?? "—",
      event.kind,
      date(event.timestamp_ns),
    ]) {
      const cell = document.createElement("td");
      if (value === event.kind) {
        const b = document.createElement("button");
        b.textContent = value;
        b.onclick = () => inspect(event);
        cell.append(b);
      } else cell.textContent = value;
      row.append(cell);
    }
    $("events").append(row);
  }
  $("empty-events").hidden = !!events.length;
  text(
    "counts",
    `${state.events.length} events · ${state.metrics.length} metrics${state.dropped ? " · older rows retained on disk" : ""}`,
  );
}
function render() {
  if (state.run) {
    text("run-title", state.run.run_id);
    text("environment", state.run.environment_id);
    text("run-status", state.run.status);
    text(
      "run-meta",
      `${state.run.kind} · ${new Date(state.run.started_at_ns / 1e6).toLocaleString()} · learning ${state.run.metadata?.learning_status ?? "unverified"}`,
    );
  }
  renderRoute();
  renderMetrics();
  renderEvents();
}
async function poll() {
  if (state.paused) {
    setTimeout(poll, 1000);
    return;
  }
  let delay = 1000;
  try {
    const runs = await api("runs");
    state.runs = [
      ...new Map(
        [...runs.runs, ...state.olderRuns].map((r) => [r.run_id, r]),
      ).values(),
    ];
    if (!state.olderRuns.length) state.nextBefore = runs.next_before;
    $("older-runs").hidden = !state.nextBefore;
    renderRuns();
    if (state.followJob) {
      const run = state.runs.find(
        (r) => r.metadata?.dashboard_job_id === state.followJob,
      );
      if (run) {
        state.followJob = null;
        selectRun(run);
      }
    }
    if (!state.run && state.runs.length) selectRun(state.runs[0]);
    if (state.run) {
      const generation = state.generation,
        run = state.run.run_id;
      const page = await api("snapshot", { run, ...state.cursor });
      if (generation !== state.generation) return;
      state.run = page.run;
      state.cursor = page.cursor;
      for (const key of ["events", "metrics"]) {
        state[key].push(...page[key]);
        if (state[key].length > 5000) {
          state.dropped += state[key].length - 5000;
          state[key] = state[key].slice(-5000);
        }
      }
      options("log", page.logs, "No log files");
      render();
      const path = $("log").value;
      if (path) {
        if (state.logPath !== path) {
          state.logOffset = null;
          state.logPath = path;
          text("log-text", "");
        }
        const data = await api("log", {
          run,
          path,
          ...(state.logOffset === null ? {} : { offset: state.logOffset }),
        });
        if (generation !== state.generation || path !== $("log").value) return;
        const view = $("log-text"),
          atEnd = view.scrollHeight - view.scrollTop - view.clientHeight < 40;
        view.textContent = (
          (data.reset ? "" : view.textContent) + data.text
        ).slice(-131072);
        state.logOffset = data.next_offset;
        if (atEnd) view.scrollTop = view.scrollHeight;
        text(
          "log-note",
          `${path} · ${data.next_offset} / ${data.size_bytes} bytes${data.reset ? " · file truncated, cursor reset" : ""} · raw process output`,
        );
        if (data.more) delay = 100;
      }
      if (page.more.events || page.more.metrics) delay = 100;
    }
    text(
      "connection",
      state.paused ? "Paused" : `Live · ${new Date().toLocaleTimeString()}`,
    );
    $("connection").className = "";
  } catch (error) {
    text("connection", `Disconnected · retrying · ${error.message}`);
    $("connection").className = "error";
    delay = 3000;
  } finally {
    setTimeout(poll, delay);
  }
}
$("pause").onclick = () => {
  state.paused = !state.paused;
  text("pause", state.paused ? "Resume live" : "Pause live");
  text(
    "connection",
    state.paused ? "Paused · data remains on disk" : "Reconnecting…",
  );
};
$("run-search").oninput = renderRuns;
$("filter").oninput = renderEvents;
$("step").oninput = renderEvents;
$("clear-step").onclick = () => {
  $("step").value = "";
  renderEvents();
};
$("plane").onchange = renderRoute;
$("route-group").onchange = renderRoute;
$("metric").onchange = renderMetrics;
$("inspect-run").onclick = () =>
  text("detail", JSON.stringify(state.run, null, 2));
$("export").onclick = () => {
  const blob = new Blob(
    [
      JSON.stringify(
        {
          schema_version: "glr.observation.v1",
          scope: "loaded_browser_window",
          partial: true,
          cursor: state.cursor,
          run: state.run,
          events: state.events,
          metrics: state.metrics,
        },
        null,
        2,
      ),
    ],
    { type: "application/json" },
  );
  const url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = `${state.run?.run_id ?? "glr"}-observations.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("older-runs").onclick = async () => {
  const button = $("older-runs");
  button.disabled = true;
  try {
    const page = await api("runs", { before: state.nextBefore });
    state.olderRuns.push(...page.runs);
    state.runs = [
      ...new Map(
        [...state.runs, ...page.runs].map((r) => [r.run_id, r]),
      ).values(),
    ];
    state.nextBefore = page.next_before;
    button.hidden = !state.nextBefore;
    renderRuns();
  } catch (e) {
    text("connection", e.message);
  } finally {
    button.disabled = false;
  }
};
window.addEventListener("glr-select-job", (event) => {
  state.followJob = event.detail.jobId;
});
poll();
