//! Synthetic loop-closure scenes for tests and the web demo.
//!
//! Model: a robot drives one lap of a loop — a circle, or a rectangular hallway
//! (`LoopShape::Square`). Its *open-loop* estimate drifts — each keyframe's pose
//! is the true (closed) pose pushed through a drift transform `D(progress)` that
//! grows from identity at the start to a large yaw + shift at the end. So the
//! estimated trajectory spirals and its far end lands away from the start (the
//! classic un-closed loop — a "too-open" square when the path is a hallway).
//!
//! The correction stored in the `GraphDelta3D` is exactly `D(progress)^-1` per
//! keyframe, so warping the open-loop cloud maps every point back onto the clean
//! loop — closing it and overlapping the two seam ends.

use crate::graph::{GraphDelta3D, Node, Transform};
use crate::math::{Quat, Vec3};
use crate::point_cloud::PointCloud2;

/// The shape of the true (closed) trajectory the robot drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopShape {
    /// A circle of the given `radius`; lidar scatters in a box around the path.
    Circle,
    /// A square hallway with corners at `(±radius, ±radius)`; lidar points hug
    /// the two corridor walls (`hallway_width` apart).
    Square,
}

/// Knobs for the synthetic loop. Defaults produce a visually clear demo scene.
#[derive(Clone, Copy, Debug)]
pub struct LoopParams {
    pub shape: LoopShape,
    pub num_nodes: usize,
    pub points_per_node: usize,
    pub radius: f64,
    /// Corridor width for `LoopShape::Square`: points sit ~`hallway_width / 2`
    /// on either side of the path. Ignored for `Circle`.
    pub hallway_width: f64,
    /// Total accumulated yaw drift (radians) at the end of the loop.
    pub drift_yaw: f64,
    /// Total accumulated translation drift at the end of the loop.
    pub drift_shift: f64,
    /// Radius of the lidar scatter sprinkled around each keyframe (for `Square`,
    /// the scatter *across and along* the wall — the wall thickness/roughness).
    pub sensor_spread: f64,
    pub time_step_s: f64,
    pub seed: u64,

    /// Number of lidar returns from a distant landmark ("tree") seen through a
    /// window in the wall. `0` disables the window/tree entirely. These points
    /// are all observed at one instant (as the robot passes the window), so they
    /// inherit that single pose's correction — a good stress test for how a far,
    /// off-trajectory feature warps.
    pub window_tree_points: usize,
    /// Loop fraction of the window (where in the lap the robot passes it).
    pub window_fraction: f64,
    /// Half-width (in loop fraction) of the gap carved in the walls at the window.
    pub window_half: f64,
    /// How far outside the wall the tree sits.
    pub tree_distance: f64,
    /// Radius of the tree's foliage blob.
    pub tree_size: f64,
}

impl Default for LoopParams {
    fn default() -> Self {
        LoopParams {
            shape: LoopShape::Circle,
            num_nodes: 48,
            points_per_node: 60,
            radius: 10.0,
            hallway_width: 6.0,
            drift_yaw: 1.1,
            drift_shift: 2.5,
            sensor_spread: 1.6,
            time_step_s: 1.0,
            seed: 1,
            window_tree_points: 0,
            window_fraction: 0.0,
            window_half: 0.0,
            tree_distance: 0.0,
            tree_size: 0.0,
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

/// The true (closed) path at a loop `fraction` in [0, 1): its center point and
/// the unit tangent (direction of travel). The circle sweeps a full turn; the
/// square walks its perimeter starting at the bottom-right corner, counterclock-
/// wise, so travel is up → left → down → right.
fn path_at(params: &LoopParams, fraction: f64) -> (Vec3, Vec3) {
    match params.shape {
        LoopShape::Circle => {
            let angle = std::f64::consts::TAU * fraction;
            let center = Vec3::new(params.radius * angle.cos(), params.radius * angle.sin(), 0.0);
            let tangent = Vec3::new(-angle.sin(), angle.cos(), 0.0);
            (center, tangent)
        }
        LoopShape::Square => square_perimeter(params.radius, fraction),
    }
}

/// A point on the perimeter of an axis-aligned square with corners at
/// `(±radius, ±radius)`, at loop `fraction` in [0, 1): its position and unit
/// tangent. Sampling the *same* fraction on differently sized squares keeps the
/// two hallway walls concentric, so their corners miter and line up.
fn square_perimeter(radius: f64, fraction: f64) -> (Vec3, Vec3) {
    let corners = [
        Vec3::new(radius, -radius, 0.0),
        Vec3::new(radius, radius, 0.0),
        Vec3::new(-radius, radius, 0.0),
        Vec3::new(-radius, -radius, 0.0),
    ];
    let scaled = fraction.rem_euclid(1.0) * 4.0;
    let quarter = scaled.floor();
    let side = (quarter as usize) % 4;
    let local = scaled - quarter;
    let start = corners[side];
    let end = corners[(side + 1) % 4];
    let center = Vec3::new(
        start.x + (end.x - start.x) * local,
        start.y + (end.y - start.y) * local,
        0.0,
    );
    let along = Vec3::new(end.x - start.x, end.y - start.y, 0.0);
    let length = (along.x * along.x + along.y * along.y).sqrt();
    let tangent = Vec3::new(along.x / length, along.y / length, 0.0);
    (center, tangent)
}

/// The lidar scatter added to a path center at a given `fraction`. The circle
/// fills a box (so interior points are visible warping). The square puts points
/// on one of the two hallway walls — a concentric square `hallway_width / 2`
/// inside or outside the path — so the walls' corners meet cleanly; a little
/// `sensor_spread` roughness thickens the wall and jitters along it.
fn scatter(params: &LoopParams, fraction: f64, center: Vec3, rng: &mut Rng) -> Vec3 {
    match params.shape {
        LoopShape::Circle => Vec3::new(
            rng.next_unit() * params.sensor_spread,
            rng.next_unit() * params.sensor_spread,
            0.0,
        ),
        LoopShape::Square => {
            let side_sign = if rng.next_01() < 0.5 { -1.0 } else { 1.0 };
            let wall_radius = params.radius + side_sign * params.hallway_width * 0.5;
            let (wall_center, wall_tangent) = square_perimeter(wall_radius, fraction);
            let perpendicular = Vec3::new(-wall_tangent.y, wall_tangent.x, 0.0);
            let across = rng.next_unit() * params.sensor_spread * 0.5;
            let along = rng.next_unit() * params.sensor_spread;
            let point = Vec3::new(
                wall_center.x + perpendicular.x * across + wall_tangent.x * along,
                wall_center.y + perpendicular.y * across + wall_tangent.y * along,
                0.0,
            );
            Vec3::new(point.x - center.x, point.y - center.y, 0.0)
        }
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
    // Maps a progress in [0, 1] onto the same loop fraction the nodes use, so a
    // point's timestamp always brackets between two keyframe nodes. The fraction
    // stops one step short of a full lap so the last node overlaps node 0 only
    // once the loop is closed (leaving a visible open seam otherwise).
    let fraction_at = |progress: f64| progress * last_index / params.num_nodes as f64;
    let time_at = |progress: f64| progress * last_index * params.time_step_s;

    // Keyframe nodes: evenly spaced around the loop. The last node sits one step
    // before the seam so it overlaps node 0 once the loop is closed.
    for index in 0..params.num_nodes {
        let progress = index as f64 / last_index;
        let fraction = fraction_at(progress);
        let time = time_at(progress);

        let (true_center, _tangent) = path_at(params, fraction);
        let drift = drift_at(params, progress);
        let open_center = apply(&drift, true_center);

        node_positions.push([open_center.x as f32, open_center.y as f32, open_center.z as f32]);
        node_times.push(time);

        graph_delta.nodes.push(Node { id: index as u64, ts: time, position: open_center });
        graph_delta.transforms.push(inverse(&drift));
    }

    // Lidar points sampled *continuously* along the loop, not snapped to a node.
    // Each point's drift is its own progress, so its correction is blended from
    // the two bracketing keyframes — the distortion is shared across neighboring
    // nodes (a sliding window) rather than jumping node-by-node.
    let total_points = params.num_nodes * params.points_per_node;
    for _ in 0..total_points {
        let progress = rng.next_01();
        let fraction = fraction_at(progress);
        let time = time_at(progress);

        // Carve a gap in the wall at the window so the opening is visible.
        if params.window_tree_points > 0 {
            let delta = (fraction - params.window_fraction).abs();
            let wrapped = delta.min(1.0 - delta);
            if wrapped < params.window_half {
                continue;
            }
        }

        let (true_center, _tangent) = path_at(params, fraction);
        let offset = scatter(params, fraction, true_center, &mut rng);
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

    // A distant tree seen through the window. Every return shares the *single*
    // instant the robot passes the window, so all of them are drifted by that one
    // pose and (once corrected) inherit that pose's node correction — showing how
    // a far, off-trajectory feature rides along with its observing keyframe.
    if params.window_tree_points > 0 {
        let window_progress = params.window_fraction * params.num_nodes as f64 / last_index;
        let window_time = time_at(window_progress);
        let window_drift = drift_at(params, window_progress);

        let (wall_center, _tangent) = path_at(params, params.window_fraction);
        let length = (wall_center.x * wall_center.x + wall_center.y * wall_center.y)
            .sqrt()
            .max(1e-9);
        let outward = Vec3::new(wall_center.x / length, wall_center.y / length, 0.0);
        let tree_center = Vec3::new(
            wall_center.x + outward.x * params.tree_distance,
            wall_center.y + outward.y * params.tree_distance,
            0.0,
        );

        for _ in 0..params.window_tree_points {
            // Uniform disk → a round canopy blob.
            let angle = rng.next_01() * std::f64::consts::TAU;
            let radius = params.tree_size * rng.next_01().sqrt();
            let true_point = Vec3::new(
                tree_center.x + radius * angle.cos(),
                tree_center.y + radius * angle.sin(),
                0.0,
            );
            let open_point = apply(&window_drift, true_point);
            cloud.points.push([open_point.x as f32, open_point.y as f32, open_point.z as f32]);
            timestamps.push(window_time);
        }
    }

    cloud.timestamps = Some(timestamps);
    SyntheticScene { cloud, graph_delta, node_positions, node_times }
}
