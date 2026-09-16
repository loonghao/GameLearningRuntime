import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { LogPanel } from "./components/ObservationPanels";
import { OutputViewer, parseOutput } from "./components/StructuredOutput";

it("renders legacy JSONL as records and exposes the omitted log prefix", async () => {
  vi.spyOn(globalThis, "fetch").mockImplementation(
    async () =>
      ({
        ok: true,
        status: 200,
        json: async () => ({
          text: 'partial previous record}\n{"kind":"status","hp":91,"money":82}\n{"kind":"opponent_roster","roster":[{"name":"凯南","star":2}]}\n',
          offset: 1000,
          next_offset: 1200,
          size_bytes: 1200,
          reset: false,
          tail_truncated: true,
          partial_start: true,
          more: false,
        }),
      }) as Response,
  );
  render(
    <LogPanel runId="run-legacy" paths={["trainer.log"]} paused={false} />,
  );
  expect(
    await screen.findByText(/Earlier bytes are not loaded/),
  ).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Earlier output" })).toBeEnabled();
  fireEvent.click(
    await screen.findByRole("button", { name: "View output record 2" }),
  );
  expect(screen.getByText("money")).toBeInTheDocument();
  expect(screen.getByText("82")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "View output record 3" }));
  fireEvent.click(screen.getByText(/roster/, { selector: "summary" }));
  expect(await screen.findByText("凯南")).toBeInTheDocument();
});

it("renders FFmpeg fields, preserves raw output and keeps markup inert", () => {
  const text =
    "\x1b[32mframe= 120 fps=30.0 time=00:00:04.00 speed=1.0x\x1b[0m\r";
  const parsed = parseOutput(text);
  expect(parsed[0].value).toEqual({
    frame: "120",
    fps: "30.0",
    time: "00:00:04.00",
    speed: "1.0x",
  });
  const { rerender } = render(<OutputViewer text={text} />);
  expect(screen.getByText("120")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Raw text" }));
  expect(screen.getByText(/frame= 120/)).toBeInTheDocument();
  rerender(
    <OutputViewer
      key="safe"
      text={'{"message":"<script>alert(1)</script>"}'}
    />,
  );
  expect(screen.getByText("<script>alert(1)</script>")).toBeInTheDocument();
  expect(document.querySelector("script")).toBeNull();
});

it("can expand every table row without silently truncating arrays", () => {
  render(
    <OutputViewer
      text={JSON.stringify(
        Array.from({ length: 65 }, (_, i) => ({ name: `unit-${i}`, score: i })),
      )}
    />,
  );
  expect(screen.queryByText("unit-64")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /Show more items/ }));
  fireEvent.click(screen.getByRole("button", { name: /Show more items/ }));
  expect(screen.getByText("unit-64")).toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: /Show more items/ }),
  ).not.toBeInTheDocument();
});

it("reconstructs a JSON record across earlier pages without mixing it with event-store records", async () => {
  const tail = '":91}\n';
  const prefix = '{"kind":"status","hp';
  const fetch = vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (url) => {
      const earlier = new URL(String(url), "http://localhost").searchParams.has(
        "before",
      );
      return {
        ok: true,
        status: 200,
        json: async () => ({
          text: earlier ? prefix : tail,
          offset: earlier ? 0 : prefix.length,
          next_offset: earlier ? prefix.length : prefix.length + tail.length,
          size_bytes: prefix.length + tail.length,
          reset: false,
          partial_start: !earlier,
          partial_end: earlier,
        }),
      } as Response;
    });
  render(<LogPanel runId="run-split" paths={["trainer.log"]} paused={false} />);
  const earlier = await screen.findByRole("button", { name: "Earlier output" });
  await waitFor(() => expect(earlier).toBeEnabled());
  fireEvent.click(earlier);
  expect(await screen.findByText("hp")).toBeInTheDocument();
  expect(screen.getByText("91")).toBeInTheDocument();
  expect(
    fetch.mock.calls.some(([url]) =>
      String(url).includes(`before=${prefix.length}`),
    ),
  ).toBe(true);
  expect(
    screen.queryByText(/Earlier bytes are not loaded/),
  ).not.toBeInTheDocument();
});
