import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  MediaWorkspace,
  frameForEvent,
  type Artifact,
} from "./components/MediaWorkspace";
import type { Event } from "./lib/api";

const run = {
  run_id: "run-media",
  environment_id: "test",
  kind: "training",
  status: "succeeded",
  started_at_ns: 1,
  metadata: {},
};
const artifact = (path: string, kind: string, role = "evidence"): Artifact => ({
  path,
  kind,
  role,
  mime: "",
  size_bytes: 100,
  available: true,
  recorded_sha256: "abc",
  integrity: "not_reverified",
});
const event: Event = {
  sequence_id: 8,
  timestamp_ns: 1,
  kind: "agent.decision",
  step_id: 4,
  episode_id: "episode-a",
  payload: {},
};
const frames = [
  { episode_id: "episode-a", step_id: 4, frame_index: 0, seconds: 1 },
  { episode_id: "episode-a", step_id: 5, frame_index: 1, seconds: 3 },
];
function serve(items: Artifact[], text = "") {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
    const path = new URL(String(url), "http://localhost").pathname;
    const value = path.endsWith("/frames")
      ? { frames, truncated: false }
      : path.endsWith("/document")
        ? { text, truncated: false }
        : { items, next_after: null };
    return { ok: true, json: async () => value } as Response;
  });
}
describe("run media", () => {
  it("does not seek a repeated step without a unique episode binding", () => {
    expect(
      frameForEvent([...frames, { ...frames[0], episode_id: "episode-b" }], {
        ...event,
        episode_id: null,
      }),
    ).toBeUndefined();
    expect(frameForEvent(frames, event)?.seconds).toBe(1);
  });
  it("links a registered video to explicit capture frames and decision selection", async () => {
    serve([
      artifact("capture.mp4", "video"),
      artifact("capture.manifest.json", "text", "capture-manifest"),
    ]);
    const onFrame = vi.fn();
    const props = {
      run,
      events: [event],
      selectedEvent: null as Event | null,
      onInspect: vi.fn(),
      onFrame,
    };
    const { rerender } = render(<MediaWorkspace {...props} />);
    const video = (await screen.findByLabelText(
      "Run recording",
    )) as HTMLVideoElement;
    await screen.findByText(/2 mapped frames/);
    video.currentTime = 3;
    fireEvent.timeUpdate(video);
    fireEvent.click(screen.getByRole("button", { name: "Inspect step 5" }));
    expect(onFrame).toHaveBeenCalledWith(frames[1]);
    rerender(<MediaWorkspace {...props} selectedEvent={event} />);
    await waitFor(() => expect(video.currentTime).toBe(1));
  });
  it("renders Markdown with registered images while keeping raw HTML and external media inert", async () => {
    serve(
      [artifact("notes.md", "markdown"), artifact("chart.png", "image")],
      "# Findings\n\n![Plot](chart.png)\n\n![External](https://outside.invalid/private)\n\n<script>alert(1)</script>\n\n[unsafe](javascript:alert(1))",
    );
    const { container } = render(
      <MediaWorkspace
        run={run}
        events={[]}
        selectedEvent={null}
        onInspect={() => {}}
        onFrame={() => {}}
      />,
    );
    await screen.findByRole("tab", { name: /Documents 1/ });
    fireEvent.mouseDown(screen.getByRole("tab", { name: /Documents/ }), {
      button: 0,
      ctrlKey: false,
    });
    await screen.findByRole("heading", { name: "Findings" });
    expect(screen.getByRole("img", { name: "Plot" })).toHaveAttribute(
      "src",
      "/api/v1/media/file?run=run-media&path=chart.png",
    );
    expect(
      screen.queryByRole("img", { name: "External" }),
    ).not.toBeInTheDocument();
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector('a[href^="javascript:"]')).toBeNull();
  });
  it("keeps local preview out of the server and revokes its URL when closed", async () => {
    const create = vi.fn(() => "blob:local-preview"),
      revoke = vi.fn();
    Object.defineProperty(URL, "createObjectURL", {
      value: create,
      configurable: true,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      value: revoke,
      configurable: true,
    });
    const fetch = serve([]);
    render(
      <MediaWorkspace
        run={run}
        events={[]}
        selectedEvent={null}
        onInspect={() => {}}
        onFrame={() => {}}
      />,
    );
    fireEvent.change(screen.getByLabelText("Preview local media"), {
      target: {
        files: [new File(["video"], "local.mp4", { type: "video/mp4" })],
      },
    });
    expect(await screen.findByText(/temporary, not saved/)).toBeInTheDocument();
    expect(screen.getByLabelText("Run recording")).toHaveAttribute(
      "src",
      "blob:local-preview",
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Close local preview" }),
    );
    expect(revoke).toHaveBeenCalledWith("blob:local-preview");
    expect(
      fetch.mock.calls.every(
        ([, init]) => !init?.method || init.method === "GET",
      ),
    ).toBe(true);
  });
});
