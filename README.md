# fast-reform

Apply bone-like deformation to point clouds quickly.

When a SLAM system detects a loop closure, pose-graph optimization (PGO) retro-
actively corrects every keyframe pose to remove accumulated drift. But the global
map — millions of accumulated lidar points — was built against the *old*, drifted
poses. `fast-reform` propagates the pose corrections onto those points so the map
stays consistent, correcting **each point by the pose-graph state at the moment it
was observed**.

| open loop (drifted) | closed loop (corrected) |
| :-----------------: | :---------------------: |
| ![open](https://raw.githubusercontent.com/jeff-hykin/fast-reform/master/docs/demo-open.png) | ![closed](https://raw.githubusercontent.com/jeff-hykin/fast-reform/master/docs/demo-closed.png) |

Drag the demo slider from 0% to 100% and the drifted cloud warps, per point, onto
the corrected pose graph. Points are colored by observation time (rainbow), so you
can watch the timeline untwist.

- **Per-point**, not time-bucketed — every point corrected at its own position and time.
- **Locally space-preserving** — the warp does not fold the cloud over itself.
- **Seam-aware** — where a loop crosses itself, points stay bound to the segment they were seen on.
- **Sparsifiable** — thin a dense pose graph to a node count *or* an error budget, with warm-started updates.
- **Parallel on native** (rayon), **serial on wasm** (raw C-ABI, no wasm-bindgen), no `rand` dependency.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Core concepts & conventions](#core-concepts--conventions)
- [How the warp works](#how-the-warp-works)
- [Reusing a correction: `Reformer`](#reusing-a-correction-reformer)
- [Thinning a dense pose graph](#thinning-a-dense-pose-graph)
- [Differences from `apply_closure.py`](#differences-from-apply_closurepy)
- [API reference](#api-reference)
- [Performance & parallelism](#performance--parallelism)
- [WebAssembly & the web demo](#webassembly--the-web-demo)
- [Building & testing](#building--testing)
- [Project status](#project-status)
- [License & attribution](#license--attribution)

## Installation

`fast-reform` is a library crate. Add it with Cargo:

```bash
cargo add fast-reform
```

or in `Cargo.toml`:

```toml
[dependencies]
fast-reform = "0.1"
```

The only native dependency is [rayon](https://docs.rs/rayon) (pulled in
automatically on non-wasm targets). Building the WebAssembly module additionally
needs the target: `rustup target add wasm32-unknown-unknown`.

## Quick start

Build a correction by hand and warp a two-point cloud:

```rust
use fast_reform::{
    apply_closure_to_cloud, GraphDelta3D, Node, PointCloud2, Quat, Transform, Vec3,
};

// Two points, each tagged with the time it was observed.
let cloud = PointCloud2 {
    points: vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
    timestamps: Some(vec![0.0, 1.0]),
    ..PointCloud2::default()
};

// A PGO correction with two keyframes: node 0 (t = 0) is already correct;
// node 1 (t = 1) needs a +2 shift in x to undo late-loop drift.
let correction = GraphDelta3D {
    nodes: vec![
        Node { id: 0, ts: 0.0, position: Vec3::new(0.0, 0.0, 0.0) },
        Node { id: 1, ts: 1.0, position: Vec3::new(10.0, 0.0, 0.0) },
    ],
    transforms: vec![
        Transform::IDENTITY,
        Transform { translation: Vec3::new(2.0, 0.0, 0.0), rotation: Quat::IDENTITY },
    ],
    // 0 = auto-pick the Gaussian bandwidths from node spacing.
    blend_sigma: 0.0,
    blend_time_sigma: 0.0,
};

let warped = apply_closure_to_cloud(&cloud, &correction);
// The early point barely moves; the late point picks up (most of) the +2 shift.
assert!(warped.points[0][0].abs() < 0.5);
assert!(warped.points[1][0] > 11.0);
```

## Core concepts & conventions

**`PointCloud2`** — the cloud carried through the warp. Only XYZ positions change;
everything else is preserved.

| field | meaning |
| --- | --- |
| `points: Vec<[f32; 3]>` | XYZ positions in world frame (the only thing the warp rewrites) |
| `timestamps: Option<Vec<f64>>` | per-point observation time in seconds; `None` = "unknown age" |
| `intensities: Option<Vec<f32>>` | carried through untouched |
| `frame_id: String`, `ts: f64` | cloud-level metadata, carried through |

**`GraphDelta3D`** — the PGO correction: equal-length `nodes` and `transforms`.

| field | meaning |
| --- | --- |
| `nodes: Vec<Node>` | keyframes — each an `id`, a `ts`, and a `position` anchor |
| `transforms: Vec<Transform>` | the SE(3) delta applied to each keyframe |
| `blend_sigma: f64` | spatial Gaussian bandwidth (`0` ⇒ auto = 2× mean nearest-neighbor spacing) |
| `blend_time_sigma: f64` | temporal Gaussian bandwidth (`0` ⇒ auto, likewise) |

**`Node`** — a keyframe. `position` is the anchor in the cloud's *pre-correction*
(open-loop) frame; the warp weights nodes by how close a point is to these anchors.
`ts` binds the node to the timeline. `id` is a **stable identity** that survives
across PGO updates, so a sparsifier can match "the nodes I kept last time" onto a
fresh correction whose deltas changed.

**`Transform`** — one SE(3) correction: `translation: Vec3` and `rotation: Quat`.

Conventions to know:

- **Quaternions are `(x, y, z, w)`** (scalar last), matching the reference impl.
- **Deltas left-multiply in the world frame:** `post = delta · pre`.
- **Rotations blend with `nlerp`** (normalized lerp), not slerp — cheaper, with
  negligible error for the small corrections loop closure produces.
- **Positions are `f32`; the warp computes in `f64`** internally for accuracy.
- **Timestamps are optional.** With them, the warp is seam-aware; without them it
  falls back to a spatial-only blend.

## How the warp works

For every point, the correction is a **proximity-weighted blend** of the node
deltas — weighted by closeness in *both space and time*:

1. **Weigh every node.** For node `i`, weight
   `w_i = exp(-d² / 2σ²) · exp(-Δt² / 2σ_t²)`, where `d` is the distance from the
   point to the node's anchor position and `Δt` is the gap between the point's
   observation time and the node's time.
2. **Blend a correction.** Normalize the weights and average the node deltas —
   `nlerp`-style weighted quaternion average on rotation, weighted mean on
   translation.
3. **Apply it rigidly.** `warped = rotation · point + translation`.

Why both terms:

- The **spatial** term makes the correction a smooth function of *position*, so
  neighboring points get near-identical corrections. The warp is therefore
  **locally space-preserving** — nearby points keep their relative arrangement and
  the cloud does not fold over itself.
- The **time** term disambiguates a loop **seam**. Where the drifted trajectory
  crosses back over itself, two segments occupy the same space but need opposite
  corrections; spatial distance alone would average them (closing nothing), so the
  time term keeps each point bound to the segment it was actually observed on.

Fallbacks: points with no timestamp use a spatial-only blend; a point far from
every node (all Gaussians underflow) takes its nearest node's rigid delta; an
empty cloud or empty correction is a pass-through.

|  thin lidar ring, open  |  thin lidar ring, closed  |
| :---------------------: | :-----------------------: |
| ![ring open](https://raw.githubusercontent.com/jeff-hykin/fast-reform/master/docs/open-loop.png) | ![ring closed](https://raw.githubusercontent.com/jeff-hykin/fast-reform/master/docs/closed-loop.png) |

The green marker is the loop's start, the red marker its end. Before closure they
sit far apart (the drift gap); applying the correction slides the two ends into
overlap.

## Reusing a correction: `Reformer`

`apply_closure_to_cloud` rebuilds the internal blend arrays (normalized rotations,
node anchors/times, resolved bandwidths) on every call. When you apply the same
correction to many clouds — or re-warp a live map as points stream in — hold a
`Reformer` instead: it owns the correction and caches those arrays, rebuilding only
when the correction changes.

```rust
use fast_reform::Reformer;

let mut reformer = Reformer::new(correction);
let warped = reformer.apply(&cloud);       // uses the cached blend

reformer.sparsify(12);                      // thin, cache rebuilt once
reformer.set_delta(next_delta);             // full PGO update, cache rebuilt once
let warped_again = reformer.apply(&cloud);
```

`Reformer` keeps both the **full** correction (what the sparsifiers thin from) and
the **active** subset (what `apply` uses). `graph_delta()` returns the active one,
`full_delta()` the full one, and `kept_ids()` the node ids currently kept.

## Thinning a dense pose graph

A real pose graph can carry far more keyframes than the warp needs — neighboring
nodes often hold nearly the same delta, so the blend would reconstruct a dropped
node's correction from its neighbors anyway.

### By node count

`sparsify` thins a correction to a target node count with **greedy leave-one-out
decimation**:

```rust
use fast_reform::sparsify;

let thinned = sparsify(&correction, 12); // keep the 12 least-redundant nodes
```

Repeatedly, for every remaining node it measures how far off the *other* nodes are
when they blend to reconstruct that node's own delta (applied at the node's
anchor), then drops the node with the smallest error — the most redundant one —
until `target_nodes` remain. Bandwidths are re-derived from the shrinking set each
round so the error uses the same blend the final sparse graph will. Nodes are
treated as an unordered set, so this also works for multi-robot graphs with no
single keyframe sequence. In the demo, thinning 48 nodes down to ~10 still closes
the loop to a ~4 px seam gap.

### By error budget

When you would rather bound the *error* than pick a count, `sparsify_within_error`
drops as many nodes as it can while the total anchor error — measured against the
**full** deformation, so the budget bounds real drift — stays within `max_error`:

```rust
use fast_reform::sparsify_within_error;

let thinned = sparsify_within_error(&correction, 0.05, None);
```

It precomputes the full-graph deformation at every node anchor once, then greedily
drops the cheapest-to-remove node while the summed anchor displacement of the
dropped nodes stays within budget.

### Warm-started updates

Each keyframe carries a stable `id`, so when a new PGO correction arrives (same
nodes, updated deltas) a `Reformer` **warm-starts** the thinning from the nodes it
kept last time instead of re-searching from scratch:

```rust
let mut reformer = Reformer::new(full_correction);
reformer.sparsify_within_error(0.05);                 // initial thin

// ... next PGO update over the same node ids ...
reformer.update_within_error(next_correction, 0.05);  // warm-started
let warped = reformer.apply(&cloud);
```

Because it begins from the previous selection (adding a few nodes back if the new
deltas pushed it over budget, or dropping a few more if there is slack), a graph
with hundreds of nodes re-thins in a handful of iterations rather than from every
node.

> **Note:** the budget is measured at node anchors, not over the whole cloud, so it
> bounds anchor drift rather than every individual point. For most maps the anchors
> are a faithful proxy; a probe-point variant can tighten this if needed.

## Differences from `apply_closure.py`

| aspect | Python original | fast-reform |
| --- | --- | --- |
| granularity | quantizes points into `time_step_s` buckets, one blend per bucket | **per-point** — every point corrected individually |
| correction field | keyed by time only (bucket) | **space + time** proximity blend — locally non-folding, seam-aware |
| rotation blend | true SLERP | **nlerp** (normalized lerp — cheaper, negligible error for small deltas) |
| sparsification | — | **greedy decimation** by node count or error budget, warm-startable |
| parallelism | numpy vectorized | **rayon** on native, serial on wasm |

## API reference

Everything below is re-exported from the crate root (`use fast_reform::...`).

### Warping

- `apply_closure_to_cloud(cloud: &PointCloud2, delta: &GraphDelta3D) -> PointCloud2`
  — warp a whole cloud; pass-through on an empty cloud or empty correction.
- `warp_positions(points: &[[f32; 3]], point_times: Option<&[f64]>, deltas: &BlendDeltas) -> Vec<[f32; 3]>`
  — warp raw positions against pre-built blend arrays (`GraphDelta3D::to_blend_arrays()`).

### Sparsification

- `sparsify(delta: &GraphDelta3D, target_nodes: usize) -> GraphDelta3D` — thin to a
  node count. `target >= len` (or an empty graph) is a pass-through.
- `sparsify_within_error(delta: &GraphDelta3D, max_error: f64, seed_ids: Option<&[u64]>) -> GraphDelta3D`
  — thin to an error budget; `seed_ids` warm-starts from a prior selection.

### `Reformer`

| method | purpose |
| --- | --- |
| `new(delta) -> Reformer` | build and precompute the blend cache (nothing thinned yet) |
| `apply(&cloud) -> PointCloud2` | warp a cloud with the cached active correction |
| `apply_positions(points, point_times) -> Vec<[f32; 3]>` | warp raw positions |
| `set_delta(delta)` | replace the full correction (resets thinning), rebuild cache |
| `sparsify(target_nodes)` | thin the full correction to a node count |
| `sparsify_within_error(max_error)` | thin to an error budget, warm-started from current kept |
| `update_within_error(delta, max_error)` | push a new correction (matched by id) and re-thin, warm-started |
| `graph_delta() -> &GraphDelta3D` | the active (possibly thinned) correction |
| `full_delta() -> &GraphDelta3D` | the full correction |
| `kept_ids() -> Vec<u64>` | ids currently kept (the next warm-start seed) |
| `blend() -> &BlendDeltas`, `is_empty() -> bool` | cache access / emptiness |

### Types

- `PointCloud2` — the cloud (see [conventions](#core-concepts--conventions)); helpers
  `len`, `is_empty`, `with_points`.
- `GraphDelta3D` — the correction; helpers `len`, `is_empty`, `scaled(amount)`
  (scale every delta between identity and full — used by the demo), `to_blend_arrays`.
- `Node { id, ts, position }`, `Transform { translation, rotation }` (`Transform::IDENTITY`,
  `scaled_from_identity(amount)`).
- `BlendDeltas` — the pre-resolved arrays the warp consumes.
- `Vec3`, `Quat` — small SE(3) math: `Quat` is `(x, y, z, w)` with `rotate`,
  `conjugate`, `normalized_or_identity`; free functions `nlerp`, `lerp_vec3`.

### Synthetic scenes (tests & demo)

- `generate_scene(&LoopParams) -> SyntheticScene` — a drifted loop plus its exact
  closure correction, fully deterministic (no `rand` dependency). Tune with
  `LoopParams` (node count, points per node, radius, drift, sensor spread, seed).

## Performance & parallelism

- The per-point loop is embarrassingly parallel and runs on **rayon** on native
  targets; the wasm build uses a serial loop (selected at compile time via
  `cfg(target_arch = "wasm32")`).
- Warping is `O(points × nodes)`. For very dense graphs, `sparsify` first to cut
  the `nodes` factor.
- Greedy decimation is `O(n³)` worst case in the node count `n`; the warm-started
  path starts near the answer, so live re-thinning of a large graph typically costs
  only a few iterations. Score-against-the-full-field targets are precomputed once
  (`O(n²)`), so the error-budget variant is no more expensive in order.
- No `unsafe` on native. The wasm module uses one mutable global for the demo
  scene (safe because wasm is single-threaded).

## WebAssembly & the web demo

[`web/`](web/) is a **no-build**, plain-JS + canvas demo. It loads the wasm module
directly (raw C-ABI, **no wasm-bindgen** — JS reads flat `f32` arrays straight out
of linear memory via `Float32Array(memory.buffer, ptr, len)`) and renders:

- the **lidar points**, rainbow-colored by observation time,
- the **pose graph** (nodes + edges, with a dashed closing edge),
- **start/end markers** so you can watch the seam overlap,
- a **closure slider** (and ▶Play) that scales the correction from 0% (open loop)
  to 100% (closed loop) in real time,
- a **nodes slider** that thins the pose graph from all keyframes down to a handful
  (via `sparsify`) so you can watch the loop still close with far fewer nodes.

The exported C-ABI is intentionally tiny (`fr_init`, `fr_warp`, `fr_max_nodes`, and
pointer/length getters in [`src/wasm_api.rs`](src/wasm_api.rs)) — a hand-rolled ABI
is simpler than a bindings toolchain for a demo that only pushes two sliders and
pulls back flat arrays.

## Building & testing

```bash
cargo test        # run the suite
cargo build --release --target wasm32-unknown-unknown   # build the wasm module

./run/build       # native lib + wasm, staged to web/fast_reform.wasm
./run/web          # rebuild wasm, serve web/, open the demo in your browser
```

Both `run/*` scripts are Deno + [dax](https://github.com/dsherret/dax). A
[`flake.nix`](flake.nix) dev shell provides `rustc`/`cargo`/`deno`.

## Project status

`0.1.x`, pre-publish. The library API (warp, `Reformer`, both sparsifiers) is in
place and covered by tests; the crate is not yet published to crates.io. Possible
next steps: a probe-point error bound over the full cloud, and optional `serde`
support for `GraphDelta3D` / `PointCloud2`.

## License & attribution

Apache-2.0, matching the upstream jnav `apply_closure` source this port is derived
from.
