import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import fixtures from "../../docs/examples/workbench-views.json";
import { parseWorkbench } from "./lib/workbench";
import { AgentWorkbench } from "./components/AgentWorkbench";
import { ProcessTrace, durationOf } from "./components/ProcessTrace";
import type { Event } from "./lib/api";
const event = (id: number, kind: string, payload = {}): Event => ({
  sequence_id: id,
  kind,
  payload,
  timestamp_ns: id * 1e9,
  step_id: id,
  episode_id: "ep-1",
  source: "bridge.test",
});
describe("game-neutral workbench", () => {
  it("accepts shared combat, economy, survival and custom fixtures and rejects invalid views", () => {
    fixtures.valid.forEach((v) => expect(parseWorkbench(v)).not.toBeNull());
    fixtures.invalid.forEach((v) => expect(parseWorkbench(v)).toBeNull());
    expect(
      parseWorkbench({
        ...fixtures.valid[0],
        sections: Array(9).fill(fixtures.valid[0].sections[0]),
      }),
    ).toBeNull();
  });
  it("switches independent producer views without mixing state and retains provenance", () => {
    const inspect = vi.fn();
    const reports = fixtures.valid.slice(0, 3).map((view, i) => ({
      ...event(i, "bridge.state", { workbench: view }),
      source: `bridge.${i}`,
    }));
    render(
      <AgentWorkbench
        run={null}
        events={[]}
        bridge={{ states: reports, truncated: false }}
        onInspect={inspect}
      />,
    );
    expect(screen.getByText("Health")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Workbench source"), {
      target: { value: "bridge.1" },
    });
    expect(screen.getByText("Gold")).toBeInTheDocument();
    expect(screen.queryByText("Health")).not.toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Inspect source report" }),
    );
    expect(inspect).toHaveBeenCalledWith(reports[1]);
    fireEvent.change(screen.getByLabelText("Workbench source"), {
      target: { value: "bridge.2" },
    });
    expect(screen.getByText("Upgrade choices")).toBeInTheDocument();
  });
  it("keeps unsupported historical views inspectable", () => {
    const report = event(1, "bridge.state", { workbench: fixtures.invalid[0] });
    const inspect = vi.fn();
    render(
      <AgentWorkbench
        run={null}
        events={[]}
        bridge={{ states: [report], truncated: true }}
        onInspect={inspect}
      />,
    );
    expect(
      screen.getByText("This reported view cannot be rendered"),
    ).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Inspect source report" }),
    );
    expect(inspect).toHaveBeenCalledWith(report);
  });
});
describe("agent process trace", () => {
  it("filters rows, opens reported input/output, and selects timeline events", () => {
    const events = [
      event(1, "agent.decision", { summary: "Choose dodge" }),
      event(2, "agent.execution", {
        duration_ms: 1250,
        input: { action: "dodge" },
        output: { accepted: true },
      }),
      event(3, "learning.update", { summary: "Checkpoint saved" }),
    ];
    function View() {
      const [selected, setSelected] = useState<Event | null>(null);
      return (
        <ProcessTrace
          run={null}
          events={events}
          selectedEvent={selected}
          onInspect={setSelected}
        />
      );
    }
    render(<View />);
    fireEvent.change(screen.getByLabelText("Process category"), {
      target: { value: "Tools" },
    });
    expect(
      screen.queryByRole("button", { name: "Open process event 1" }),
    ).not.toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Open process event 2" }),
    );
    expect(screen.getByLabelText("Input")).toHaveTextContent(
      '"action": "dodge"',
    );
    expect(screen.getByLabelText("Output / receipt")).toHaveTextContent(
      '"accepted": true',
    );
    expect(screen.getByText("1.3s")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Close process detail"));
    expect(
      screen.queryByLabelText("Process event details"),
    ).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Process category"), {
      target: { value: "All" },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Trace Model event 1" }),
    );
    expect(screen.getByLabelText("Process event details")).toHaveTextContent(
      "Choose dodge",
    );
    fireEvent.change(screen.getByLabelText("Search process"), {
      target: { value: "Checkpoint" },
    });
    expect(
      screen.getByRole("button", { name: "Open process event 3" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Open process event 2" }),
    ).not.toBeInTheDocument();
  });
  it("never infers duration from neighboring events or invalid reported values", () => {
    for (const payload of [
      {},
      { duration_ms: -1 },
      { duration_ms: "120" },
      { duration_ms: Infinity },
    ])
      expect(durationOf(event(1, "agent.execution", payload))).toBeNull();
    expect(durationOf(event(1, "agent.execution", { duration_ms: 0 }))).toBe(0);
  });
});
