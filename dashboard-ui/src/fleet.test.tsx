import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { FleetWorkspace } from "./components/FleetWorkspace";
import App from "./App";
import type {
  FleetConsumerReceipt,
  FleetDataset,
  FleetMachine,
  FleetSnapshot,
  FleetView,
} from "./lib/api";

const NOW = Date.UTC(2031, 2, 14, 9, 26, 53);
const utc = (secondsAgo: number) =>
  new Date(NOW - secondsAgo * 1000).toISOString();
const machine = (overrides: Partial<FleetMachine> = {}): FleetMachine => ({
  source_id: "fixture-source-c",
  source_epoch: "fixture-epoch-3",
  machine_id: "fixture-machine-7",
  simulated: false,
  revoked: false,
  declared_status: "running",
  run_id: "fixture-run-8",
  game_id: "fixture-game-2",
  environment_id: "fixture.environment-2",
  source_revision: "test-rev-4",
  runtime_source_commit: "d".repeat(40),
  adapter_source_sha256: "e".repeat(64),
  behavior_policy_sha256: "f".repeat(64),
  checkpoint_sha256: null,
  heartbeat_declared_at_utc: utc(12),
  heartbeat_received_at_utc: utc(10),
  data_observed_at_utc: utc(300),
  data_received_at_utc: utc(40),
  ...overrides,
});
const snapshot = (overrides: Partial<FleetSnapshot> = {}): FleetSnapshot => ({
  schema_version: "glr.fleet.snapshot.v1",
  generated_at_utc: utc(5),
  scope: "local_trusted_sources",
  heartbeat_ttl_seconds: 120,
  machines: [machine()],
  datasets: [],
  consumer_receipts: [],
  ...overrides,
});
const available = (value: FleetSnapshot): FleetView => ({
  schema_version: "glr.fleet.view.v1",
  status: "available",
  snapshot: value,
});
const response = (value: unknown, ok = true) =>
  ({ ok, status: ok ? 200 : 503, json: async () => value }) as Response;
function mockView(value: unknown) {
  return vi.spyOn(globalThis, "fetch").mockResolvedValue(response(value));
}
async function loaded(value: FleetSnapshot = snapshot()) {
  const fetch = mockView(available(value));
  render(<FleetWorkspace />);
  await screen.findByRole("table", { name: "Machines and runs" });
  return fetch;
}
beforeEach(() => {
  vi.spyOn(Date, "now").mockReturnValue(NOW);
});
describe("fleet evidence", () => {
  it("reports a missing snapshot as unknown and only uses a read-only GET", async () => {
    const fetch = mockView({
      schema_version: "glr.fleet.view.v1",
      status: "missing",
      snapshot: null,
    });
    render(<FleetWorkspace />);
    expect(
      await screen.findByText(
        "No fleet snapshot recorded. Machine activity is unknown.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText("Online (reported)")).not.toBeInTheDocument();
    expect(screen.queryByText(/Generated UTC/)).not.toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(String(fetch.mock.calls[0][0])).toBe("/api/v1/fleet?");
    expect(fetch.mock.calls[0][1]?.method).toBeUndefined();
  });
  it("keeps heartbeat received and both data clocks independent", async () => {
    await loaded();
    const heartbeat = screen.getByLabelText(
      "fixture-source-c heartbeat received",
    );
    const observed = screen.getByLabelText("fixture-source-c data observed");
    const received = screen.getByLabelText("fixture-source-c data received");
    expect(within(heartbeat).getByText("10s ago")).toBeInTheDocument();
    expect(within(observed).getByText("5m ago")).toBeInTheDocument();
    expect(within(observed).getByText("Stale")).toBeInTheDocument();
    expect(within(received).getByText("40s ago")).toBeInTheDocument();
    expect(within(heartbeat).getByText(utc(10))).toBeInTheDocument();
    expect(within(observed).getByText(utc(300))).toBeInTheDocument();
    expect(within(received).getByText(utc(40))).toBeInTheDocument();
    expect(screen.getByText("Online (reported)")).toBeInTheDocument();
  });
  it("vetoes online for a stale snapshot despite a recent heartbeat", async () => {
    await loaded(snapshot({ generated_at_utc: utc(121) }));
    expect(screen.getAllByText("Snapshot stale").length).toBeGreaterThan(0);
    expect(screen.queryByText("Online (reported)")).not.toBeInTheDocument();
  });
  it("preserves the exact freshness boundary", async () => {
    await loaded(
      snapshot({
        generated_at_utc: utc(120),
        machines: [
          machine({
            heartbeat_declared_at_utc: utc(120),
            heartbeat_received_at_utc: utc(120),
          }),
        ],
      }),
    );
    expect(screen.getByText("Online (reported)")).toBeInTheDocument();
    vi.mocked(Date.now).mockReturnValue(NOW + 1);
    fireEvent.click(screen.getByRole("button", { name: "Refresh snapshot" }));
    await waitFor(() =>
      expect(screen.queryByText("Online (reported)")).not.toBeInTheDocument(),
    );
    expect(screen.getAllByText("Snapshot stale").length).toBeGreaterThan(0);
  });
  it.each([
    "heartbeat_declared_at_utc",
    "heartbeat_received_at_utc",
    "data_observed_at_utc",
    "data_received_at_utc",
  ] as const)(
    "treats future %s as clock mismatch instead of fresh",
    async (field) => {
      await loaded(snapshot({ machines: [machine({ [field]: utc(-1) })] }));
      expect(screen.getAllByText("Clock mismatch").length).toBeGreaterThan(0);
      expect(screen.queryByText("Online (reported)")).not.toBeInTheDocument();
    },
  );
  it("vetoes online when generated snapshot time is future", async () => {
    await loaded(snapshot({ generated_at_utc: utc(-1) }));
    expect(screen.getByText("Snapshot clock mismatch")).toBeInTheDocument();
    expect(screen.queryByText("Online (reported)")).not.toBeInTheDocument();
  });
  it("marks simulated sources even with running and fresh timestamps", async () => {
    await loaded(snapshot({ machines: [machine({ simulated: true })] }));
    expect(screen.getAllByText("SIMULATED").length).toBeGreaterThan(1);
    expect(screen.queryByText("Online (reported)")).not.toBeInTheDocument();
  });
  it.each([
    {
      name: "unattached",
      generated: utc(5),
      fields: {
        heartbeat_declared_at_utc: null,
        heartbeat_received_at_utc: null,
      },
      expected: "SIMULATED · Unknown",
    },
    {
      name: "stale snapshot",
      generated: utc(121),
      fields: {},
      expected: "SIMULATED · Snapshot stale",
    },
    {
      name: "stale heartbeat",
      generated: utc(5),
      fields: { heartbeat_received_at_utc: utc(121) },
      expected: "SIMULATED · Heartbeat stale",
    },
    {
      name: "future source clock",
      generated: utc(5),
      fields: { heartbeat_received_at_utc: utc(-1) },
      expected: "SIMULATED · Clock mismatch",
    },
  ])(
    "shows $name evidence independently from simulated mode",
    async ({ generated, fields, expected }) => {
      await loaded(
        snapshot({
          generated_at_utc: generated,
          machines: [machine({ simulated: true, ...fields })],
        }),
      );
      const row = screen.getByText("fixture-machine-7").closest("tr")!;
      expect(within(row).getByText(expected)).toBeInTheDocument();
      expect(within(row).getByText("SIMULATED")).toBeInTheDocument();
      expect(
        within(row).queryByText("Online (reported)"),
      ).not.toBeInTheDocument();
    },
  );
  it("does not infer activity for registered sources without heartbeat", async () => {
    await loaded(
      snapshot({
        machines: [
          machine({
            heartbeat_declared_at_utc: null,
            heartbeat_received_at_utc: null,
          }),
        ],
      }),
    );
    expect(
      within(
        screen.getByRole("table", { name: "Machines and runs" }),
      ).getAllByText("Unknown").length,
    ).toBeGreaterThan(0);
    expect(screen.queryByText("Online (reported)")).not.toBeInTheDocument();
  });
  it.each([
    { revoked: true, expected: "Revoked" },
    { heartbeat_received_at_utc: utc(121), expected: "Heartbeat stale" },
  ])(
    "does not show revoked or stale-heartbeat sources online",
    async ({ expected, ...fields }) => {
      await loaded(snapshot({ machines: [machine(fields)] }));
      expect(screen.getByText(expected)).toBeInTheDocument();
      expect(screen.queryByText("Online (reported)")).not.toBeInTheDocument();
    },
  );
  it("does not manufacture heartbeat time or freshness on repeated reads", async () => {
    const fetch = await loaded();
    vi.mocked(Date.now).mockReturnValue(NOW + 130000);
    fireEvent.click(screen.getByRole("button", { name: "Refresh snapshot" }));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    await waitFor(() =>
      expect(screen.queryByText("Online (reported)")).not.toBeInTheDocument(),
    );
    expect(
      within(
        screen.getByLabelText("fixture-source-c heartbeat received"),
      ).getByText(utc(10)),
    ).toBeInTheDocument();
    expect(
      fetch.mock.calls.every(([, init]) => init?.method === undefined),
    ).toBe(true);
  });
  it("shows distinct compatibility cohorts, holdout, rejection and revocation without claiming consumption", async () => {
    const dataset: FleetDataset = {
      source_id: "fixture-source-c",
      source_epoch: "fixture-epoch-3",
      compatibility_group_sha256: "1".repeat(64),
      assignment_id: "holdout-set",
      split: "evaluation_holdout",
      quality_status: "ready",
      accepted_transition_count: 17,
      duplicate_transition_count: 4,
      rejected_shard_count: 2,
      ready_shard_count: 3,
      retained_bytes: 211,
      last_plan_eligible: false,
      last_plan_reason_codes: ["holdout", "policy_mismatch"],
    };
    await loaded(
      snapshot({
        datasets: [
          dataset,
          {
            ...dataset,
            compatibility_group_sha256: "2".repeat(64),
            assignment_id: "revoked-set",
            split: "train",
            quality_status: "revoked",
            last_plan_reason_codes: ["revoked"],
          },
        ],
      }),
    );
    const table = screen.getByRole("table", {
      name: "Dataset compatibility and quality",
    });
    expect(within(table).getByText("1".repeat(64))).toBeInTheDocument();
    expect(within(table).getByText("2".repeat(64))).toBeInTheDocument();
    expect(within(table).getByText("Evaluation holdout")).toBeInTheDocument();
    expect(within(table).getAllByText(/2 rejected/).length).toBe(2);
    expect(within(table).getAllByText("revoked").length).toBeGreaterThan(0);
    expect(
      within(table).getByText("holdout, policy mismatch"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "No durable consumer receipts in this snapshot. Consumption is unknown.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText("Consumed (recorded)")).not.toBeInTheDocument();
  });
  it("renders historic receipt scope without joining a current epoch or run", async () => {
    const receipt: FleetConsumerReceipt = {
      receipt_id: "historical-receipt",
      plan_id: "historical-plan",
      learner_id: "historical-learner",
      source_ids: ["historical-source-not-in-current-machines"],
      status: "consumed",
      transition_count: 23,
      callback_completed: true,
      learner_declared_updates: 0,
      finished_at_utc: utc(70),
    };
    await loaded(
      snapshot({
        consumer_receipts: [
          receipt,
          {
            ...receipt,
            receipt_id: "unknown-receipt",
            status: "unknown_effect",
            callback_completed: false,
            learner_declared_updates: null,
            finished_at_utc: null,
          },
        ],
      }),
    );
    const table = screen.getByRole("table", {
      name: "Durable consumer receipts",
    });
    expect(
      within(table).getAllByText("historical-source-not-in-current-machines")
        .length,
    ).toBe(2);
    expect(within(table).getByText("Consumed (recorded)")).toBeInTheDocument();
    expect(within(table).getByText("Effect unknown")).toBeInTheDocument();
    expect(within(table).getByText("0")).toBeInTheDocument();
    expect(within(table).getByText("Observed completed")).toBeInTheDocument();
    expect(
      within(table).getByText("Not observed completed"),
    ).toBeInTheDocument();
    expect(within(table).queryByText("fixture-run-8")).not.toBeInTheDocument();
    expect(
      screen.getByText(/Epoch scope is held in the durable plan/),
    ).toBeInTheDocument();
  });
  it.each(["invalid", "unavailable"])(
    "uses safe fixed state for %s",
    async (status) => {
      mockView({ schema_version: "glr.fleet.view.v1", status, snapshot: null });
      render(<FleetWorkspace />);
      expect(
        await screen.findByText(
          status === "invalid"
            ? "Fleet snapshot rejected. Machine activity is unknown."
            : "Fleet snapshot unavailable. Machine activity is unknown.",
        ),
      ).toBeInTheDocument();
      expect(screen.queryByText("Online (reported)")).not.toBeInTheDocument();
    },
  );
  it("does not expose fetch or raw invalid-view details", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new Error("private-example-detail-must-stay-hidden"),
    );
    render(<FleetWorkspace />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Fleet snapshot unavailable",
    );
    expect(
      screen.queryByText(/private-example-detail/),
    ).not.toBeInTheDocument();
  });
  it("expires online at the snapshot deadline without fetching or changing source times", async () => {
    vi.useFakeTimers({
      toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval"],
    });
    try {
      const fetch = mockView(
        available(snapshot({ generated_at_utc: utc(119.999) })),
      );
      const view = render(<FleetWorkspace />);
      await act(async () => {
        await Promise.resolve();
      });
      expect(screen.getByText("Online (reported)")).toBeInTheDocument();
      vi.mocked(Date.now).mockReturnValue(NOW + 2);
      act(() => vi.advanceTimersByTime(3));
      expect(screen.queryByText("Online (reported)")).not.toBeInTheDocument();
      expect(fetch).toHaveBeenCalledTimes(1);
      expect(
        within(
          screen.getByLabelText("fixture-source-c heartbeat received"),
        ).getByText(utc(10)),
      ).toBeInTheDocument();
      view.unmount();
    } finally {
      vi.useRealTimers();
    }
  });
  it("aborts its own read on unmount", async () => {
    const fetch = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(() => new Promise(() => {}));
    const view = render(<FleetWorkspace />);
    expect(fetch).toHaveBeenCalledTimes(1);
    const signal = fetch.mock.calls[0][1]?.signal;
    view.unmount();
    expect(signal?.aborted).toBe(true);
  });
  it("adds Fleet to the existing read-only workbench navigation", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
      const path = new URL(String(url), "http://localhost").pathname;
      if (path.endsWith("/health"))
        return response({
          read_only: true,
          environment_id: "fixture.ui",
          version: "fixture-version",
        });
      if (path.endsWith("/runs"))
        return response({ runs: [], next_before: null });
      if (path.endsWith("/fleet"))
        return response({
          schema_version: "glr.fleet.view.v1",
          status: "missing",
          snapshot: null,
        });
      throw new Error("unexpected fixture request");
    });
    render(<App />);
    const tab = screen.getByRole("tab", { name: "Fleet" });
    fireEvent.mouseDown(tab, { button: 0, ctrlKey: false });
    expect(
      await screen.findByRole("region", { name: "Fleet workspace" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("tab", { name: "Training & operations" }),
    ).not.toBeInTheDocument();
  });
});
