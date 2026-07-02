//! Synthetic loop-closure scenes for tests and the web demo.
//!
//! Model: a robot drives one lap of a circle. Its *open-loop* estimate drifts —
//! each keyframe's pose is the true (closed) pose pushed through a drift
//! transform `D(progress)` that grows from identity at the start to a large yaw +
//! shift at the end. So the estimated trajectory spirals and its far end lands
//! away from the start (the classic un-closed loop).
//!
//! The correction stored in the `GraphDelta3D` is exactly `D(progress)^-1` per
//! keyframe, so warping the open-loop cloud maps every point back onto the clean
//! circle — closing the loop and overlapping the two seam ends.

use crate::graph::{GraphDelta3D, Node, Transform};
use crate::math::{Quat, Vec3};
use crate::point_cloud::PointCloud2;

/// Knobs for the synthetic loop. Defaults produce a visually clear demo scene.
#[derive(Clone, Copy, Debug)]
pub struct LoopParams {
    pub num_nodes: usize,
    pub points_per_node: usize,
    pub radius: f64,
    /// Total accumulated yaw drift (radians) at the end of the loop.
    pub drift_yaw: f64,
    /// Total accumulated translation drift at the end of the loop.
    pub drift_shift: f64,
    /// Radius of the lidar scatter sprinkled around each keyframe.
    pub sensor_spread: f64,
    pub time_step_s: f64,
    pub seed: u64,
}

impl Default for LoopParams {
    fn default() -> Self {
        LoopParams {
            num_nodes: 48,
            points_per_node: 60,
            radius: 10.0,
            drift_yaw: 1.1,
            drift_shift: 2.5,
            sensor_spread: 1.6,
            time_step_s: 1.0,
            seed: 1,
        }
    }
}

/// A generated scene: the open-loop cloud, the closure correction, and the
/// open-loop pose-graph node positions (for drawing nodes + edges).
pub struct SyntheticScene {
    pub cloud: PointCloud2,
    pub graph_delta: GraphDelta3D,
    pub node_positions: Vec<[f32; 3]>,
    pub node_times: Vec<f64>,
}

/// Deterministic small PRNG (SplitMix64) so scenes are reproducible without a
/// `rand` dependency (keeps the wasm build light).
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15) }
    }

    fn next_01(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Top 53 bits → [0, 1).
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_unit(&mut self) -> f64 {
        self.next_01() * 2.0 - 1.0
    }
}

/// Drift transform applied to true poses at a given progress in [0, 1].
/// Rotation about +Z by `drift_yaw * progress`, plus a small linear shift.
fn drift_at(params: &LoopParams, progress: f64) -> Transform {
    let yaw = params.drift_yaw * progress;
    let half = yaw * 0.5;
    Transform {
        rotation: Quat::new(0.0, 0.0, half.sin(), half.cos()),
        translation: Vec3::new(params.drift_shift * progress, 0.0, 0.0),
    }
}

fn apply(transform: &Transform, point: Vec3) -> Vec3 {
    let rotated = transform.rotation.rotate(point);
    Vec3::new(
        rotated.x + transform.translation.x,
        rotated.y + transform.translation.y,
        rotated.z + transform.translation.z,
    )
}

/// Inverse of a drift transform: the correction that undoes it.
/// For `p' = R p + t`, the inverse is `p = R^-1 p' - R^-1 t`.
fn inverse(transform: &Transform) -> Transform {
    let inverse_rotation = transform.rotation.conjugate();
    let shifted = inverse_rotation.rotate(transform.translation);
    Transform {
        rotation: inverse_rotation,
        translation: Vec3::new(-shifted.x, -shifted.y, -shifted.z),
    }
}

pub fn generate_scene(params: &LoopParams) -> SyntheticScene {
    let mut rng = Rng::new(params.seed);

    let mut cloud = PointCloud2 {
        frame_id: "map".to_string(),
        ts: 0.0,
        ..PointCloud2::default()
    };
    let mut timestamps: Vec<f64> = Vec::new();

    let mut graph_delta = GraphDelta3D::default();
    let mut node_positions: Vec<[f32; 3]> = Vec::new();
    let mut node_times: Vec<f64> = Vec::new();

    let last_index = (params.num_nodes.max(2) - 1) as f64;
    let two_pi = std::f64::consts::TAU;
    // Maps a progress in [0, 1] onto the same angle/time schedule the nodes use,
    // so a point's timestamp always brackets between two keyframe nodes.
    let angle_at = |progress: f64| two_pi * progress * last_index / params.num_nodes as f64;
    let time_at = |progress: f64| progress * last_index * params.time_step_s;

    // Keyframe nodes: evenly spaced around the loop. The last node sits one step
    // before the seam so it overlaps node 0 once the loop is closed.
    for index in 0..params.num_nodes {
        let progress = index as f64 / last_index;
        let angle = angle_at(progress);
        let time = time_at(progress);

        let true_center = Vec3::new(params.radius * angle.cos(), params.radius * angle.sin(), 0.0);
        let drift = drift_at(params, progress);
        let open_center = apply(&drift, true_center);

        node_positions.push([open_center.x as f32, open_center.y as f32, open_center.z as f32]);
        node_times.push(time);

        graph_delta.nodes.push(Node { ts: time, position: open_center });
        graph_delta.transforms.push(inverse(&drift));
    }

    // Lidar points sampled *continuously* along the loop, not snapped to a node.
    // Each point's drift is its own progress, so its correction is blended from
    // the two bracketing keyframes — the distortion is shared across neighboring
    // nodes (a sliding window) rather than jumping node-by-node.
    let total_points = params.num_nodes * params.points_per_node;
    for _ in 0..total_points {
        let progress = rng.next_01();
        let angle = angle_at(progress);
        let time = time_at(progress);

        let true_center = Vec3::new(params.radius * angle.cos(), params.radius * angle.sin(), 0.0);
        let offset = Vec3::new(
            rng.next_unit() * params.sensor_spread,
            rng.next_unit() * params.sensor_spread,
            0.0,
        );
        let true_point = Vec3::new(
            true_center.x + offset.x,
            true_center.y + offset.y,
            true_center.z + offset.z,
        );
        let drift = drift_at(params, progress);
        let open_point = apply(&drift, true_point);
        cloud.points.push([open_point.x as f32, open_point.y as f32, open_point.z as f32]);
        timestamps.push(time);
    }

    cloud.timestamps = Some(timestamps);
    SyntheticScene { cloud, graph_delta, node_positions, node_times }
}
