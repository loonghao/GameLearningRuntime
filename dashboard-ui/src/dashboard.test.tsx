import {
  act,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor,
} from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import App from "./App";
import { TrainingControls } from "./components/TrainingControls";
import { BridgePanel, RoutePanel } from "./components/ObservationPanels";
import { operationArgv, type Event, type Operation } from "./lib/api";
import { useObservation } from "./lib/use-observation";

const run = (id = "run-example") => ({
  run_id: id,
  environment_id: "test.react",
  kind: "training",
  status: "running",
  started_at_ns: 1,
  metadata: {},
});
const event: Event = {
  sequence_id: 1,
  timestamp_ns: 1,
  kind: "navigation.route_sample",
  step_id: 4,
  episode_id: "episode-1",
  payload: { position: [1, 2, 3] },
};
const job = {
  id: "job-example",
  created_at_ms: 1,
  status: "running",
  requested_argv: ["train"],
  exit_code: null,
};
const response = (value: unknown, ok = true) =>
  ({ ok, status: ok ? 200 : 503, json: async () => value }) as Response;
const preset = {
  schema_version: "glr.training-preset.v1",
  id: "train.default",
  title: "Default training",
  description: "Synthetic test configuration",
  argv: ["train"],
};
function mocks(
  readOnly = false,
  post?: (init: RequestInit) => Promise<Response>,
) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (url, init) => {
    const path = new URL(String(url), "http://localhost").pathname;
    if (path.endsWith("/health"))
      return response({
        read_only: readOnly,
        environment_id: "test.react",
        version: "test",
      });
    if (path.endsWith("/runs"))
      return response({ runs: [run()], next_before: null });
    if (path.endsWith("/snapshot"))
      return response({
        run: run(),
        events: [event],
        metrics: [],
        logs: [],
        cursor: { events_after: 1, metrics_after: 0 },
      });
    if (path.endsWith("/telemetry/state"))
      return response({ states: [], truncated: false });
    if (path.endsWith("/presets")) return response({ data: [preset] });
    if (path.endsWith("/catalog"))
      return response({
        data: {
          commands: [{ path: ["train"], description: "Train", arguments: [] }],
        },
      });
    if (path.endsWith("/jobs") && init?.method === "POST")
      return post ? post(init) : response({ data: job });
    if (path.endsWith("/jobs"))
      return response({ data: { jobs: [], next_before: null } });
    if (path.endsWith("/job-log"))
      return response({
        data: { text: "Persisted output", tail_truncated: false },
      });
    throw new Error(`Unexpected request ${url}`);
  });
}
describe("shared command controls", () => {
  it("preserves repeated flags, positional multiple values, and spaces as argv", () => {
    const operation = {
      path: ["task", "run"],
      arguments: [
        { id: "name", long: null, takes_value: true, multiple: false },
        {
          id: "set",
          long: "set",
          takes_value: true,
          multiple: true,
          repeat: true,
        },
        { id: "check", long: "check", takes_value: false },
      ],
    } as Operation;
    expect(
      operationArgv(operation, {
        name: "my task",
        set: "budget=4\nname=a b",
        check: true,
      }),
    ).toEqual([
      "task",
      "run",
      "my task",
      "--set",
      "budget=4",
      "--set",
      "name=a b",
      "--check",
    ]);
  });
  it("submits a preset once even on repeated clicks, then follows the durable job", async () => {
    let resolve!: (r: Response) => void;
    const post = vi.fn(
      () =>
        new Promise<Response>((r) => {
          resolve = r;
        }),
    );
    mocks(false, post);
    const select = vi.fn();
    render(<TrainingControls onSelectJob={select} />);
    const button = await screen.findByRole("button", {
      name: "Start training",
    });
    fireEvent.click(button);
    fireEvent.click(button);
    expect(post).toHaveBeenCalledTimes(1);
    await act(async () => resolve(response({ data: job })));
    await waitFor(() => expect(select).toHaveBeenCalledWith("job-example"));
    expect(await screen.findByText("Persisted output")).toBeInTheDocument();
  });
  it("reuses the mutation request ID after an uncertain response", async () => {
    const post = vi
      .fn()
      .mockRejectedValueOnce(new Error("response lost"))
      .mockResolvedValueOnce(response({ data: job }));
    mocks(false, post);
    render(<TrainingControls onSelectJob={() => {}} />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Start training" }),
    );
    await screen.findByRole("alert");
    fireEvent.click(screen.getByRole("button", { name: "Start training" }));
    await waitFor(() => expect(post).toHaveBeenCalledTimes(2));
    expect(JSON.parse(post.mock.calls[0][0].body).request_id).toBe(
      JSON.parse(post.mock.calls[1][0].body).request_id,
    );
  });
});
describe("observation views", () => {
  it("keeps observation mode read-only and exposes the selected event payload", async () => {
    mocks(true);
    render(<App />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Inspect event 1" }),
    );
    expect(
      screen.queryByRole("tab", { name: /Training & operations/ }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: "Filter step" })).toHaveValue(
      4,
    );
    expect(screen.getByText(/"position":/)).toBeInTheDocument();
  });
  it("displays source-bound progress with received time and an inspect action", () => {
    const inspect = vi.fn();
    render(
      <BridgePanel
        bridge={{
          states: [
            {
              ...event,
              source: "bridge.test",
              kind: "bridge.progress",
              payload: { fraction: 0.4, label: "Loading map" },
            },
          ],
          truncated: false,
        }}
        onInspect={inspect}
      />,
    );
    expect(screen.getByRole("progressbar")).toHaveAttribute(
      "aria-valuenow",
      "40",
    );
    fireEvent.click(screen.getByRole("button", { name: /Loading map/ }));
    expect(inspect).toHaveBeenCalledWith(
      expect.objectContaining({ source: "bridge.test" }),
    );
    expect(screen.getByText(/^Reported /)).toBeInTheDocument();
  });
  it("scrubs route samples without merging different episodes", () => {
    const inspect = vi.fn();
    render(
      <RoutePanel
        events={[
          event,
          {
            ...event,
            sequence_id: 2,
            step_id: 5,
            payload: { position: [3, 4, 5] },
          },
          { ...event, sequence_id: 3, episode_id: "episode-other" },
        ]}
        onInspect={inspect}
      />,
    );
    fireEvent.change(screen.getByRole("slider"), { target: { value: "1" } });
    expect(inspect).toHaveBeenLastCalledWith(
      expect.objectContaining({ sequence_id: 2 }),
    );
    expect(screen.getByText(/2 samples/)).toBeInTheDocument();
  });
  it("ignores a late response from a previously selected run", async () => {
    let resolve!: (r: Response) => void;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
      const query = new URL(String(url), "http://localhost"),
        id = query.searchParams.get("run")!;
      if (query.pathname.endsWith("state"))
        return response({ states: [], truncated: false });
      if (id === "run-old")
        return new Promise<Response>((r) => {
          resolve = r;
        });
      return response({
        run: run(id),
        events: [],
        metrics: [],
        logs: [],
        cursor: { events_after: -1, metrics_after: 0 },
      });
    });
    const { result, rerender } = renderHook(
      ({ id }) => useObservation(id, false),
      { initialProps: { id: "run-old" } },
    );
    rerender({ id: "run-new" });
    await waitFor(() => expect(result.current.run?.run_id).toBe("run-new"));
    await act(async () =>
      resolve(
        response({
          run: run("run-old"),
          events: [event],
          metrics: [],
          logs: [],
          cursor: { events_after: 1, metrics_after: 0 },
        }),
      ),
    );
    expect(result.current.run?.run_id).toBe("run-new");
    expect(result.current.events).toHaveLength(0);
  });
});
