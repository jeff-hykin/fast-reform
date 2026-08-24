# fast-reform

Apply bone-like deformation to point clouds quickly.

**[▶ Live demo](https://jeff-hykin.github.io/fast-reform/)** — drag the slider and watch the drifted cloud warp, per point, onto the corrected pose graph.

| open loop (drifted) | closed loop (corrected) |
| :-----------------: | :---------------------: |
| ![open](https://raw.githubusercontent.com/jeff-hykin/fast-reform/master/docs/demo-open.png) | ![closed](https://raw.githubusercontent.com/jeff-hykin/fast-reform/master/docs/demo-closed.png) |

A pose graph gives you one SE(3) correction per keyframe. `fast-reform` spreads
those corrections across the points themselves, correcting **each point by the
graph state at the moment it was observed**.

- **Per-point**, not time-bucketed.
- **Locally space-preserving** — the warp does not fold the cloud over itself.
- **Seam-aware** — where a loop crosses itself, points stay bound to the segment they were seen on.
- **Sparsifiable** — thin a dense graph to a node count *or* an error budget, warm-startable.
- **Parallel on native** (rayon), **serial on wasm** (raw C-ABI, no wasm-bindgen), no `rand` dependency.

## Installation

```bash
cargo add fast-reform
```

The only native dependency is [rayon](https://docs.rs/rayon). The wasm build needs
`rustup target add wasm32-unknown-unknown`.

## Quick start

```rust
use fast_reform::{
    reform, GraphDelta3D, Node, PointCloud2, Quat, Transform, Vec3,
};

// Two points, each tagged with the time it was observed.
let cloud = PointCloud2 {
    points: vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
    timestamps: Some(vec![0.0, 1.0]),
    ..PointCloud2::default()
};

// Two keyframes: node 0 (t = 0) is already correct; node 1 (t = 1) needs a
// +2 shift in x to undo late drift.
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

let warped = reform(&cloud, &correction);
// The early point barely moves; the late point picks up (most of) the +2 shift.
assert!(warped.points[0][0].abs() < 0.5);
assert!(warped.points[1][0] > 11.0);
```

## How the warp works

For every point, the correction is a proximity-weighted blend of the node deltas,
weighted by closeness in **both space and time**:

```
w_i = exp(-d² / 2σ²) · exp(-Δt² / 2σ_t²)
```

Normalize the weights, average the deltas (`nlerp` on rotation, weighted mean on
translation), apply rigidly. The spatial term keeps neighboring points moving
together; the time term disambiguates a **seam**, where the drifted trajectory
crosses back over itself and two overlapping segments need opposite corrections.

Fallbacks: no timestamp ⇒ spatial-only blend; a point far from every node takes
its nearest node's delta; empty cloud or empty correction is a pass-through.

| thin lidar ring, open | thin lidar ring, closed |
| :-------------------: | :---------------------: |
| ![ring open](https://raw.githubusercontent.com/jeff-hykin/fast-reform/master/docs/open-loop.png) | ![ring closed](https://raw.githubusercontent.com/jeff-hykin/fast-reform/master/docs/closed-loop.png) |

## Types

**`PointCloud2`** — only XYZ positions change; everything else is carried through.

| field | meaning |
| --- | --- |
| `points: Vec<[f32; 3]>` | XYZ in world frame (the only thing the warp rewrites) |
| `timestamps: Option<Vec<f64>>` | per-point observation time in seconds; `None` = "unknown age" |
| `intensities: Option<Vec<f32>>` | carried through untouched |
| `frame_id: String`, `ts: f64` | cloud-level metadata |

**`GraphDelta3D`** — the correction: equal-length `nodes` and `transforms`.

| field | meaning |
| --- | --- |
| `nodes: Vec<Node>` | keyframes — each an `id`, a `ts`, and a `position` anchor |
| `transforms: Vec<Transform>` | the SE(3) delta applied to each keyframe |
| `blend_sigma: f64` | spatial Gaussian bandwidth (`0` ⇒ auto = 2× mean nearest-neighbor spacing) |
| `blend_time_sigma: f64` | temporal Gaussian bandwidth (`0` ⇒ auto) |

`Node.position` is the anchor in the *pre-correction* frame. `Node.id` is a stable
identity that survives PGO updates, which is what makes warm-started thinning work.

Conventions: quaternions are `(x, y, z, w)`; deltas left-multiply in the world
frame (`post = delta · pre`); positions are `f32` but the warp computes in `f64`.

## Reusing a correction: `Reformer`

`reform` rebuilds its blend arrays on every call. To apply one correction to many
clouds, hold a `Reformer` — it caches those arrays and rebuilds only when the
correction changes.

```rust
use fast_reform::Reformer;

let mut reformer = Reformer::new(correction);
let warped = reformer.apply(&cloud);       // uses the cached blend

reformer.sparsify(12);                      // thin, cache rebuilt once
reformer.set_delta(next_delta);             // full PGO update, cache rebuilt once
let warped_again = reformer.apply(&cloud);
```

## Thinning a dense pose graph

Neighboring nodes often hold nearly the same delta, so the blend reconstructs a
dropped node's correction from its neighbors anyway.

```rust
use fast_reform::{sparsify, sparsify_within_error};

let thinned = sparsify(&correction, 12);                  // keep 12 least-redundant nodes
let thinned = sparsify_within_error(&correction, 0.05, None);  // or bound the error
```

`sparsify` greedily drops the most redundant node — the one its neighbors
reconstruct best — until `target_nodes` remain. `sparsify_within_error` instead
drops as many as it can while total anchor error against the **full** deformation
stays within budget. In the demo, thinning 48 nodes to ~10 still closes the loop
to a ~4 px seam gap.

Because each node has a stable `id`, a `Reformer` warm-starts the next thinning
from what it kept last time:

```rust
let mut reformer = Reformer::new(full_correction);
reformer.sparsify_within_error(0.05);                 // initial thin

// ... next PGO update over the same node ids ...
reformer.update_within_error(next_correction, 0.05);  // warm-started
let warped = reformer.apply(&cloud);
```

> **Note:** the budget is measured at node anchors, not over the whole cloud, so it
> bounds anchor drift rather than every individual point.

## API reference

Everything is re-exported from the crate root (`use fast_reform::...`).

**Warping**

- `reform(cloud: &PointCloud2, delta: &GraphDelta3D) -> PointCloud2`
- `warp_positions(points: &[[f32; 3]], point_times: Option<&[f64]>, deltas: &BlendDeltas) -> Vec<[f32; 3]>`
  — warp raw positions against pre-built arrays (`GraphDelta3D::to_blend_arrays()`).

**Sparsification**

- `sparsify(delta, target_nodes) -> GraphDelta3D` — `target >= len` is a pass-through.
- `sparsify_within_error(delta, max_error, seed_ids) -> GraphDelta3D` — `seed_ids` warm-starts.

**`Reformer`**

| method | purpose |
| --- | --- |
| `new(delta)` | build and precompute the blend cache |
| `apply(&cloud) -> PointCloud2` | warp a cloud with the cached active correction |
| `apply_positions(points, point_times)` | warp raw positions |
| `set_delta(delta)` | replace the full correction (resets thinning) |
| `sparsify(target_nodes)` | thin to a node count |
| `sparsify_within_error(max_error)` | thin to an error budget, warm-started |
| `update_within_error(delta, max_error)` | push a new correction (matched by id) and re-thin |
| `graph_delta()` / `full_delta()` | the active (thinned) / full correction |
| `kept_ids() -> Vec<u64>` | ids currently kept (the next warm-start seed) |

**Other types** — `Node`, `Transform` (`IDENTITY`, `scaled_from_identity`),
`BlendDeltas`, `Vec3`, `Quat` (`rotate`, `conjugate`, `normalized_or_identity`),
free functions `nlerp` / `lerp_vec3`. `GraphDelta3D::scaled(amount)` scales every
delta between identity and full (used by the demo). `generate_scene(&LoopParams)`
builds a deterministic drifted loop plus its exact closure correction.

## Performance

Warping is `O(points × nodes)` and embarrassingly parallel (rayon on native,
serial on wasm). For dense graphs, `sparsify` first to cut the `nodes` factor.
Greedy decimation is `O(n³)` worst case, but the warm-started path starts near the
answer and typically costs a few iterations. No `unsafe` on native.

## Web demo

[`web/`](web/) is a no-build, plain-JS + canvas demo that loads the wasm module
directly and renders the lidar points (rainbow-colored by observation time), the
pose graph, start/end seam markers, a closure slider, and a nodes slider that
thins the graph live. The exported C-ABI is intentionally tiny — see
[`src/wasm_api.rs`](src/wasm_api.rs).

## Building & testing

```bash
cargo test        # run the suite
./run/build       # native lib + wasm, staged to web/fast_reform.wasm
./run/web         # rebuild wasm, serve web/, open the demo
```

Both `run/*` scripts are Deno + [dax](https://github.com/dsherret/dax). A
[`flake.nix`](flake.nix) dev shell provides `rustc`/`cargo`/`deno`.

## License

Apache-2.0.
