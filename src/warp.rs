//! The per-point warp. Each point's correction is a proximity-weighted blend of
//! the pose-graph node deltas, using **both** spatial distance (so neighboring
//! points get near-identical corrections — the map stays locally space-preserving
//! and does not fold over itself) **and** observation-time distance (so at a loop
//! seam, where the drifted trajectory crosses back over itself, a point stays
//! bound to the segment it was actually seen on instead of averaging the two).
//! Native builds run the per-point loop in parallel with rayon; wasm builds run
//! it serially.

use crate::graph::{BlendDeltas, GraphDelta3D};
use crate::math::{Quat, Vec3};
use crate::point_cloud::PointCloud2;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// The blended (rotation, translation) correction for a point at `point` observed
/// at `time`: a partition-of-unity of the node deltas weighted by
/// `exp(-d^2 / 2 sigma^2) * exp(-dt^2 / 2 time_sigma^2)`. `time = None` drops the
/// time term (spatial blend only).
fn blend_at(deltas: &BlendDeltas, point: Vec3, time: Option<f64>) -> (Quat, Vec3) {
    let count = deltas.len();
    if count == 1 {
        return (deltas.rotations[0], deltas.translations[0]);
    }

    let inverse_two_sigma_squared = 1.0 / (2.0 * deltas.sigma * deltas.sigma);
    let inverse_two_time_sigma_squared = 1.0 / (2.0 * deltas.time_sigma * deltas.time_sigma);

    let mut weights = vec![0.0_f64; count];
    let mut nearest = 0usize;
    let mut nearest_weight = f64::NEG_INFINITY;
    let mut weight_sum = 0.0;
    for index in 0..count {
        let dx = point.x - deltas.positions[index].x;
        let dy = point.y - deltas.positions[index].y;
        let dz = point.z - deltas.positions[index].z;
        let distance_squared = dx * dx + dy * dy + dz * dz;
        let mut exponent = distance_squared * inverse_two_sigma_squared;
        if let Some(observed_at) = time {
            let dt = observed_at - deltas.times[index];
            exponent += dt * dt * inverse_two_time_sigma_squared;
        }
        let weight = (-exponent).exp();
        if weight > nearest_weight {
            nearest_weight = weight;
            nearest = index;
        }
        weights[index] = weight;
        weight_sum += weight;
    }

    // Far from every node the Gaussians underflow to ~0; fall back to the closest
    // node's rigid delta so the point still moves smoothly with its neighbors.
    if !(weight_sum > 1e-12) {
        return (deltas.rotations[nearest], deltas.translations[nearest]);
    }

    // Weighted-average the quaternions (nlerp-style), aligning the double-cover
    // sign of each to the nearest (highest-weight) node before summing.
    let reference = deltas.rotations[nearest];
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_z = 0.0;
    let mut sum_w = 0.0;
    let mut translation = Vec3::ZERO;
    for index in 0..count {
        let weight = weights[index] / weight_sum;
        let rotation = deltas.rotations[index];
        let dot = rotation.x * reference.x
            + rotation.y * reference.y
            + rotation.z * reference.z
            + rotation.w * reference.w;
        let signed = if dot < 0.0 { -weight } else { weight };
        sum_x += signed * rotation.x;
        sum_y += signed * rotation.y;
        sum_z += signed * rotation.z;
        sum_w += signed * rotation.w;
        translation.x += weight * deltas.translations[index].x;
        translation.y += weight * deltas.translations[index].y;
        translation.z += weight * deltas.translations[index].z;
    }

    let rotation = Quat::new(sum_x, sum_y, sum_z, sum_w).normalized_or_identity();
    (rotation, translation)
}

fn warp_one(point: [f32; 3], time: Option<f64>, deltas: &BlendDeltas) -> [f32; 3] {
    let position = Vec3::new(point[0] as f64, point[1] as f64, point[2] as f64);
    let (rotation, translation) = blend_at(deltas, position, time);
    let rotated = rotation.rotate(position);
    [
        (rotated.x + translation.x) as f32,
        (rotated.y + translation.y) as f32,
        (rotated.z + translation.z) as f32,
    ]
}

/// Warp every position by the correction blended at its own location and time.
/// `point_times`, when present, must align with `points`.
pub fn warp_positions(
    points: &[[f32; 3]],
    point_times: Option<&[f64]>,
    deltas: &BlendDeltas,
) -> Vec<[f32; 3]> {
    if points.is_empty() || deltas.is_empty() {
        return points.to_vec();
    }

    let time_of = |index: usize| point_times.map(|times| times[index]);

    #[cfg(not(target_arch = "wasm32"))]
    {
        (0..points.len())
            .into_par_iter()
            .map(|index| warp_one(points[index], time_of(index), deltas))
            .collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..points.len())
            .map(|index| warp_one(points[index], time_of(index), deltas))
            .collect()
    }
}

/// Warp a whole cloud by a pose-graph correction. Pass-through when the correction
/// has no nodes or the cloud is empty. Per-point timestamps (when present) keep
/// points bound to the trajectory segment they were observed on.
pub fn apply_closure_to_cloud(cloud: &PointCloud2, graph_delta: &GraphDelta3D) -> PointCloud2 {
    if graph_delta.is_empty() || cloud.is_empty() {
        return cloud.clone();
    }

    let deltas = graph_delta.to_blend_arrays();
    let warped = warp_positions(&cloud.points, cloud.timestamps.as_deref(), &deltas);
    cloud.with_points(warped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Node, Transform};

    fn translate_x(amount: f64) -> Transform {
        Transform { translation: Vec3::new(amount, 0.0, 0.0), rotation: Quat::IDENTITY }
    }

    /// Two anchors 10 apart on x: node A at origin translates +0, node B at
    /// (10,0,0) translates +6. Small sigma so each anchor dominates near itself.
    fn two_node_delta() -> GraphDelta3D {
        GraphDelta3D {
            nodes: vec![
                Node { id: 0, ts: 0.0, position: Vec3::ZERO },
                Node { id: 1, ts: 1.0, position: Vec3::new(10.0, 0.0, 0.0) },
            ],
            transforms: vec![translate_x(0.0), translate_x(6.0)],
            blend_sigma: 2.0,
            blend_time_sigma: 1.0,
        }
    }

    #[test]
    fn correction_matches_the_nearest_anchor() {
        let deltas = two_node_delta().to_blend_arrays();
        // Right on anchor A (time 0) → ~A's delta (no shift).
        let (_, at_a) = blend_at(&deltas, Vec3::ZERO, Some(0.0));
        assert!(at_a.x.abs() < 0.1, "near A expected ~0, got {}", at_a.x);
        // Right on anchor B (time 1) → ~B's delta (+6).
        let (_, at_b) = blend_at(&deltas, Vec3::new(10.0, 0.0, 0.0), Some(1.0));
        assert!((at_b.x - 6.0).abs() < 0.1, "near B expected ~6, got {}", at_b.x);
    }

    #[test]
    fn blend_is_monotonic_and_non_folding_between_anchors() {
        let deltas = two_node_delta().to_blend_arrays();
        // The output x-coordinate must stay strictly increasing as the input x
        // sweeps A→B (time interpolated with it): that is "no fold-over" locally.
        let mut previous = f64::NEG_INFINITY;
        let mut x = 0.0;
        while x <= 10.0 {
            let time = x / 10.0;
            let (_, translation) = blend_at(&deltas, Vec3::new(x, 0.0, 0.0), Some(time));
            let output = x + translation.x;
            assert!(output > previous, "folded at x={x}: {output} <= {previous}");
            previous = output;
            x += 0.25;
        }
    }

    #[test]
    fn empty_cloud_passes_through() {
        let cloud = PointCloud2::default();
        let delta = GraphDelta3D {
            nodes: vec![Node { id: 0, ts: 0.0, position: Vec3::ZERO }],
            transforms: vec![Transform::IDENTITY],
            blend_sigma: 1.0,
            blend_time_sigma: 1.0,
        };
        assert_eq!(apply_closure_to_cloud(&cloud, &delta), cloud);
    }

    #[test]
    fn empty_delta_passes_through() {
        let cloud = PointCloud2 {
            points: vec![[1.0, 2.0, 3.0]],
            ..PointCloud2::default()
        };
        let delta = GraphDelta3D::default();
        assert_eq!(apply_closure_to_cloud(&cloud, &delta), cloud);
    }

    #[test]
    fn single_node_applies_its_rigid_delta_everywhere() {
        let cloud = PointCloud2 {
            points: vec![[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]],
            ..PointCloud2::default()
        };
        let delta = GraphDelta3D {
            nodes: vec![Node { id: 0, ts: 0.0, position: Vec3::ZERO }],
            transforms: vec![translate_x(5.0)],
            blend_sigma: 1.0,
            blend_time_sigma: 1.0,
        };
        let warped = apply_closure_to_cloud(&cloud, &delta);
        assert!((warped.points[0][0] - 5.0).abs() < 1e-6);
        assert!((warped.points[1][0] - 55.0).abs() < 1e-6);
    }
}
