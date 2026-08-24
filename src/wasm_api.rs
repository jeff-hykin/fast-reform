//! Raw C-ABI wasm surface for the web demo. No wasm-bindgen: the module keeps a
//! single scene in a global, and JS reads results straight out of linear memory
//! via the exported pointers (`Float32Array(memory.buffer, ptr, len)`).
//!
//! The demo only ever pushes one input (the slider `alpha`) and pulls flat f32
//! arrays back, so a hand-rolled ABI is simpler than a bindings toolchain.

use crate::point_cloud::PointCloud2;
use crate::sparsify::sparsify;
use crate::synthetic::{generate_scene, LoopParams, LoopShape, SyntheticScene};
use crate::warp::reform;

struct DemoState {
    scene: SyntheticScene,
    /// Warped point positions as x, y interleaved (2 * point count).
    point_xy: Vec<f32>,
    /// Per-point normalized time in [0, 1] for the rainbow gradient (point count).
    point_time: Vec<f32>,
    /// Warped node positions as x, y interleaved (2 * node count).
    node_xy: Vec<f32>,
}

// wasm is single-threaded, so a mutable global is safe in practice.
static mut STATE: Option<DemoState> = None;

fn state() -> &'static mut DemoState {
    // SAFETY: single-threaded wasm; STATE is set by fr_init before any getter.
    unsafe {
        let pointer = std::ptr::addr_of_mut!(STATE);
        (*pointer).as_mut().expect("fr_init must be called first")
    }
}

/// Build the synthetic scene and warp it at alpha = 0 (open loop). `shape` picks
/// the trajectory: 0 = circle (disk of lidar), 1 = square hallway (points on the
/// two corridor walls, drifted "too open"). Returns the number of points.
#[no_mangle]
pub extern "C" fn fr_init(seed: u32, shape: u32) -> u32 {
    let params = if shape == 1 {
        // Square hallway: two corridor walls a corridor-width apart, and a
        // stronger drift so the loop clearly fails to close (a "too-open" square).
        LoopParams {
            shape: LoopShape::Square,
            seed: seed as u64,
            radius: 10.0,
            hallway_width: 6.0,
            sensor_spread: 1.5,
            points_per_node: 120,
            drift_yaw: 1.1,
            drift_shift: 4.0,
            // A window in the right wall with a tree far outside it.
            window_tree_points: 500,
            window_fraction: 0.12,
            window_half: 0.02,
            tree_distance: 16.0,
            tree_size: 3.5,
            ..LoopParams::default()
        }
    } else {
        // Circle: wide spread so points fill the disk (including the middle) — the
        // demo is about *seeing* how interior points warp, not a thin lidar ring.
        LoopParams {
            shape: LoopShape::Circle,
            seed: seed as u64,
            sensor_spread: 8.0,
            points_per_node: 90,
            ..LoopParams::default()
        }
    };
    let scene = generate_scene(&params);

    let point_count = scene.cloud.len();
    let times = scene.cloud.timestamps.clone().unwrap_or_default();
    let max_time = times.iter().cloned().fold(0.0_f64, f64::max).max(1.0);
    let point_time = times.iter().map(|&t| (t / max_time) as f32).collect();

    let demo = DemoState {
        scene,
        point_xy: vec![0.0; point_count * 2],
        point_time,
        node_xy: Vec::new(),
    };
    // SAFETY: single-threaded wasm.
    unsafe {
        let pointer = std::ptr::addr_of_mut!(STATE);
        *pointer = Some(demo);
    }
    let full_nodes = state().scene.graph_delta.len() as u32;
    fr_warp(0.0, full_nodes);
    point_count as u32
}

/// Re-warp the scene: thin the pose graph to `node_count` nodes (greedy
/// leave-one-out decimation), scale the remaining correction by `alpha`
/// (0 = open loop, 1 = fully closed loop), and refresh the output buffers. The
/// drawn nodes are the ones the sparsifier kept, warped like everything else.
#[no_mangle]
pub extern "C" fn fr_warp(alpha: f32, node_count: u32) {
    let demo = state();
    let thinned = sparsify(&demo.scene.graph_delta, node_count as usize);
    let scaled = thinned.scaled(alpha as f64);

    let warped_cloud = reform(&demo.scene.cloud, &scaled);
    for (index, point) in warped_cloud.points.iter().enumerate() {
        demo.point_xy[index * 2] = point[0];
        demo.point_xy[index * 2 + 1] = point[1];
    }

    let node_cloud = PointCloud2 {
        points: thinned
            .nodes
            .iter()
            .map(|node| [node.position.x as f32, node.position.y as f32, node.position.z as f32])
            .collect(),
        timestamps: Some(thinned.nodes.iter().map(|node| node.ts).collect()),
        ..PointCloud2::default()
    };
    let warped_nodes = reform(&node_cloud, &scaled);
    demo.node_xy.clear();
    for point in &warped_nodes.points {
        demo.node_xy.push(point[0]);
        demo.node_xy.push(point[1]);
    }
}

/// The full (un-thinned) node count — the max for the demo's node-count slider.
#[no_mangle]
pub extern "C" fn fr_max_nodes() -> u32 {
    state().scene.graph_delta.len() as u32
}

#[no_mangle]
pub extern "C" fn fr_num_points() -> u32 {
    state().point_time.len() as u32
}

#[no_mangle]
pub extern "C" fn fr_num_nodes() -> u32 {
    (state().node_xy.len() / 2) as u32
}

#[no_mangle]
pub extern "C" fn fr_points_ptr() -> *const f32 {
    state().point_xy.as_ptr()
}

#[no_mangle]
pub extern "C" fn fr_point_times_ptr() -> *const f32 {
    state().point_time.as_ptr()
}

#[no_mangle]
pub extern "C" fn fr_node_points_ptr() -> *const f32 {
    state().node_xy.as_ptr()
}
