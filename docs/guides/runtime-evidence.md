# Runtime evidence and artifact lineage

GLR exposes evidence records for adapters that need to correlate live runtime
state with training artifacts. These contracts do not control a game and do
not replace authoritative reward or terminal state.

```python
from uuid import UUID
from game_learning_runtime import (
    ArtifactLineage,
    ModalNavigationBoundary,
    ModalState,
    RouteHealthTelemetry,
    RouteTransitionEvidence,
)

edge = RouteTransitionEvidence(
    episode_id=episode_id,
    route_id="route.alpha",
    edge_id="edge.03",
    map_sha256=map_sha256,
    game_image_sha256=game_image_sha256,
    producer_state_seq=producer_state_seq,
    settled=authoritative_settled,
    reached=authoritative_reached,
    position=settled_position,
    yaw_deg=settled_yaw_deg,
    heading_error_deg=heading_error_deg,
)
if edge.edge_succeeded:
    # Adapter may advance its own route state. GLR does not issue movement.
    pass
```

`edge_succeeded` is true only for a settled, reached edge. The adapter must
obtain position/orientation and `producer_state_seq` from authoritative state;
GLR does not infer them from elapsed time, displacement alone, or screenshots.
Angles use `[-180, 180)` and heading error uses `[0, 180]`, so wraparound is
represented without a special recovery rule.

Use `RouteHealthTelemetry` as read-only diagnostics. A non-zero
`stall_ticks` or `oscillation_count` means the adapter should re-observe and
apply its own reviewed recovery policy. The record does not authorize a turn,
teleport, retry, or generic input.

Navigation should remain paused for `ModalState.OPEN` and for any non-
authoritative boundary. Only an authoritative `ModalState.CLOSED` with a
higher producer sequence clears the pause:

```python
boundary = ModalNavigationBoundary(
    episode_id=episode_id,
    modal_id="vendor",
    state=ModalState.CLOSED,
    producer_state_seq=close_seq,
    authoritative=True,
    previous=open_boundary,
)
assert not boundary.navigation_paused
```

Bind recordings and trajectories with `ArtifactLineage`. Keep `encounter_id`
opaque and stable; do not use a display name as identity. The hashes should be
the exact map/build image inputs used by the adapter. Persist the lineage as a
sidecar next to the transition or recording and include it in the model bundle
inputs. This association does not prove a successful encounter or terminal
outcome; those remain runtime-authoritative signals.
