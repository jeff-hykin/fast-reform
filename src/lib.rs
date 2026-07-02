//! fast-reform — per-point loop-closure warp for point clouds.
//!
//! A Rust reimplementation of jnav's `apply_closure`, differing deliberately:
//! each point's correction is a proximity-weighted blend of the pose-graph node
//! deltas in **both space and time** (locally non-folding, seam-aware, no
//! time-bucketing), rotations blend with nlerp, and a dense graph can be thinned
//! to a target node count with greedy leave-one-out decimation. The native build
//! parallelizes the per-point loop with rayon; the wasm build runs it serially
//! and drives the web demo.

pub mod graph;
pub mod math;
pub mod point_cloud;
pub mod reform;
pub mod sparsify;
pub mod synthetic;
pub mod warp;

#[cfg(target_arch = "wasm32")]
pub mod wasm_api;

pub use graph::{GraphDelta3D, Node, Transform};
pub use math::{Quat, Vec3};
pub use point_cloud::PointCloud2;
pub use reform::Reformer;
pub use sparsify::sparsify;
pub use warp::{apply_closure_to_cloud, warp_positions};
