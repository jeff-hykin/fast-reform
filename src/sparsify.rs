//! Node-delta sparsification: thin a dense pose-graph correction down to a target
//! number of nodes while preserving the warp it induces.
//!
//! Real pose graphs can have far more keyframes than the warp actually needs —
//! neighboring nodes often carry nearly the same delta, so the space+time blend
//! would reconstruct a dropped node's correction from its neighbors anyway. This
//! removes that redundancy with **greedy leave-one-out decimation**: repeatedly
//! drop the single node whose own delta is best predicted by blending the ones
//! that remain (the most redundant node), until only `target_nodes` are left.
//!
//! Nodes are treated as an unordered set (no trajectory assumed), so this also
//! works for multi-robot graphs where the keyframes have no single sequence.

use crate::graph::{GraphDelta3D, Node, Transform};
use crate::math::{Quat, Vec3};

/// Return a copy of `graph_delta` thinned to at most `target_nodes` nodes.
///
/// Keeps the nodes/transforms whose combined delta field best matches the
/// original; `target_nodes >= node count` (or an empty graph) is a pass-through.
/// Both blend bandwidths are carried through unchanged.
pub fn sparsify(graph_delta: &GraphDelta3D, target_nodes: usize) -> GraphDelta3D {
    let count = graph_delta.len();
    let target = target_nodes.max(1);
    if count <= target {
        return graph_delta.clone();
    }

    let ids: Vec<u64> = graph_delta.nodes.iter().map(|node| node.id).collect();
    let positions: Vec<Vec3> = graph_delta.nodes.iter().map(|node| node.position).collect();
    let times: Vec<f64> = graph_delta.nodes.iter().map(|node| node.ts).collect();
    let rotations: Vec<Quat> = graph_delta
        .transforms
        .iter()
        .map(|transform| transform.rotation.normalized_or_identity())
        .collect();
    let translations: Vec<Vec3> =
        graph_delta.transforms.iter().map(|transform| transform.translation).collect();

    let mut kept: Vec<usize> = (0..count).collect();

    while kept.len() > target {
        // Bandwidths adapt to the *current* (thinning) node set so the blend the
        // error uses matches the blend the final, sparser graph will use.
        let sigma = resolved_sigma(graph_delta.blend_sigma, &kept, &positions, spatial_distance);
        let time_sigma =
            resolved_sigma(graph_delta.blend_time_sigma, &kept, &times, |a, b| (a - b).abs());

        let mut most_redundant = 0usize;
        let mut smallest_error = f64::INFINITY;
        for slot in 0..kept.len() {
            let error = leave_one_out_error(
                slot,
                &kept,
                &positions,
                &times,
                &rotations,
                &translations,
                sigma,
                time_sigma,
            );
            if error < smallest_error {
                smallest_error = error;
                most_redundant = slot;
            }
        }
        kept.remove(most_redundant);
    }

    let nodes: Vec<Node> = kept
        .iter()
        .map(|&index| Node { id: ids[index], ts: times[index], position: positions[index] })
        .collect();
    let transforms: Vec<Transform> = kept
        .iter()
        .map(|&index| Transform { translation: translations[index], rotation: rotations[index] })
        .collect();

    GraphDelta3D {
        nodes,
        transforms,
        blend_sigma: graph_delta.blend_sigma,
        blend_time_sigma: graph_delta.blend_time_sigma,
    }
}

/// The node arrays a sparsification pass reads: parallel per-node data plus the
/// (possibly auto) blend bandwidths carried from the source `GraphDelta3D`.
struct NodeArrays {
    ids: Vec<u64>,
    positions: Vec<Vec3>,
    times: Vec<f64>,
    rotations: Vec<Quat>,
    translations: Vec<Vec3>,
    blend_sigma: f64,
    blend_time_sigma: f64,
}

impl NodeArrays {
    fn from_delta(graph_delta: &GraphDelta3D) -> Self {
        NodeArrays {
            ids: graph_delta.nodes.iter().map(|node| node.id).collect(),
            positions: graph_delta.nodes.iter().map(|node| node.position).collect(),
            times: graph_delta.nodes.iter().map(|node| node.ts).collect(),
            rotations: graph_delta
                .transforms
                .iter()
                .map(|transform| transform.rotation.normalized_or_identity())
                .collect(),
            translations: graph_delta.transforms.iter().map(|transform| transform.translation).collect(),
            blend_sigma: graph_delta.blend_sigma,
            blend_time_sigma: graph_delta.blend_time_sigma,
        }
    }

    /// Resolve the space + time bandwidths for a given member subset (auto-picks
    /// from spacing when the source configured `<= 0`), matching `to_blend_arrays`.
    fn sigmas(&self, members: &[usize]) -> (f64, f64) {
        let sigma = resolved_sigma(self.blend_sigma, members, &self.positions, spatial_distance);
        let time_sigma =
            resolved_sigma(self.blend_time_sigma, members, &self.times, |a, b| (a - b).abs());
        (sigma, time_sigma)
    }

    /// Where the blend of `members` sends the anchor at node `index`. `skip_slot`
    /// (an index *into `members`*) drops that one member from the blend; pass
    /// `usize::MAX` to blend over all of them.
    fn deform(&self, members: &[usize], skip_slot: usize, index: usize, sigma: f64, time_sigma: f64) -> Vec3 {
        let anchor = self.positions[index];
        let (rotation, translation) = blend_excluding(
            skip_slot,
            members,
            &self.positions,
            &self.times,
            &self.rotations,
            &self.translations,
            anchor,
            self.times[index],
            sigma,
            time_sigma,
        );
        apply(rotation, translation, anchor)
    }
}

/// Thin `graph_delta` as far as possible while the total reconstruction error of
/// the kept correction stays within `max_error`, measured against the **full**
/// correction at the original node anchors (not just the shrinking kept set — so
/// the budget bounds drift from the true deformation, not from an intermediate
/// one). Error is the summed anchor displacement between the full and thinned
/// deformations over the dropped nodes.
///
/// `seed_ids`, when `Some`, **warm-starts** from a previously kept selection
/// (matched by node id): the pass begins from that subset instead of the full
/// graph, adds nodes back if the seed is over budget, then keeps dropping while
/// under budget. Re-thinning an updated pose graph you have thinned before is
/// then cheap — it starts near the answer instead of from every node.
pub fn sparsify_within_error(
    graph_delta: &GraphDelta3D,
    max_error: f64,
    seed_ids: Option<&[u64]>,
) -> GraphDelta3D {
    let count = graph_delta.len();
    if count <= 1 {
        return graph_delta.clone();
    }
    let arrays = NodeArrays::from_delta(graph_delta);
    let all: Vec<usize> = (0..count).collect();

    // The "truth" every kept subset is scored against: the full-graph deformation
    // at each node anchor, computed once with the full-set bandwidths.
    let (full_sigma, full_time_sigma) = arrays.sigmas(&all);
    let full_targets: Vec<Vec3> =
        (0..count).map(|i| arrays.deform(&all, usize::MAX, i, full_sigma, full_time_sigma)).collect();

    let mut in_kept = vec![true; count];
    if let Some(seed) = seed_ids {
        let wanted: std::collections::HashSet<u64> = seed.iter().copied().collect();
        let mut any = false;
        for index in 0..count {
            in_kept[index] = wanted.contains(&arrays.ids[index]);
            any |= in_kept[index];
        }
        // A seed that matches nothing (e.g. first run, or all ids changed) falls
        // back to the full graph.
        if !any {
            in_kept.iter_mut().for_each(|flag| *flag = true);
        }
    }
    let mut kept: Vec<usize> = (0..count).filter(|&index| in_kept[index]).collect();

    // Total error of the current selection: sum over dropped nodes of how far the
    // kept blend misses the full deformation at that node's anchor.
    let total_error = |kept: &[usize], in_kept: &[bool], sigma: f64, time_sigma: f64| -> f64 {
        (0..count)
            .filter(|&d| !in_kept[d])
            .map(|d| spatial_distance(full_targets[d], arrays.deform(kept, usize::MAX, d, sigma, time_sigma)))
            .sum()
    };

    // If the warm-start seed is already over budget, add back the worst-missed
    // dropped node until it fits (or nothing is left to restore).
    loop {
        let (sigma, time_sigma) = arrays.sigmas(&kept);
        let mut worst: Option<usize> = None;
        let mut worst_error = f64::NEG_INFINITY;
        let mut total = 0.0;
        for d in 0..count {
            if in_kept[d] {
                continue;
            }
            let error =
                spatial_distance(full_targets[d], arrays.deform(&kept, usize::MAX, d, sigma, time_sigma));
            total += error;
            if error > worst_error {
                worst_error = error;
                worst = Some(d);
            }
        }
        if total <= max_error {
            break;
        }
        match worst {
            Some(d) => {
                in_kept[d] = true;
                kept = (0..count).filter(|&index| in_kept[index]).collect();
            }
            None => break,
        }
    }

    // Greedily drop the cheapest-to-remove node while the resulting total error
    // stays within budget.
    while kept.len() > 1 {
        let (sigma, time_sigma) = arrays.sigmas(&kept);
        let mut best_slot = 0;
        let mut best_error = f64::INFINITY;
        for slot in 0..kept.len() {
            let node = kept[slot];
            let reconstructed = arrays.deform(&kept, slot, node, sigma, time_sigma);
            let error = spatial_distance(full_targets[node], reconstructed);
            if error < best_error {
                best_error = error;
                best_slot = slot;
            }
        }

        let candidate = kept[best_slot];
        let mut trial = kept.clone();
        trial.remove(best_slot);
        in_kept[candidate] = false;
        let (trial_sigma, trial_time_sigma) = arrays.sigmas(&trial);
        if total_error(&trial, &in_kept, trial_sigma, trial_time_sigma) <= max_error {
            kept = trial;
        } else {
            in_kept[candidate] = true;
            break;
        }
    }

    build_delta(graph_delta, &arrays, &kept)
}

/// Assemble a `GraphDelta3D` from the kept member indices, preserving node ids and
/// carrying the source bandwidths through.
fn build_delta(source: &GraphDelta3D, arrays: &NodeArrays, kept: &[usize]) -> GraphDelta3D {
    let nodes: Vec<Node> = kept
        .iter()
        .map(|&index| Node {
            id: arrays.ids[index],
            ts: arrays.times[index],
            position: arrays.positions[index],
        })
        .collect();
    let transforms: Vec<Transform> = kept
        .iter()
        .map(|&index| Transform { translation: arrays.translations[index], rotation: arrays.rotations[index] })
        .collect();
    GraphDelta3D {
        nodes,
        transforms,
        blend_sigma: source.blend_sigma,
        blend_time_sigma: source.blend_time_sigma,
    }
}

/// How far off the remaining nodes are when they try to reconstruct the delta of
/// the node at `kept[skip_slot]`: the distance between where that node's own
/// delta sends its anchor and where the blend of the *other* kept nodes sends it.
/// Small error ⇒ the node is redundant and safe to drop.
fn leave_one_out_error(
    skip_slot: usize,
    kept: &[usize],
    positions: &[Vec3],
    times: &[f64],
    rotations: &[Quat],
    translations: &[Vec3],
    sigma: f64,
    time_sigma: f64,
) -> f64 {
    let target = kept[skip_slot];
    let anchor = positions[target];

    let actual = apply(rotations[target], translations[target], anchor);
    let (blended_rotation, blended_translation) = blend_excluding(
        skip_slot,
        kept,
        positions,
        times,
        rotations,
        translations,
        anchor,
        times[target],
        sigma,
        time_sigma,
    );
    let reconstructed = apply(blended_rotation, blended_translation, anchor);
    spatial_distance(actual, reconstructed)
}

/// The space+time proximity blend of the kept node deltas at (`point`, `time`),
/// excluding `kept[skip_slot]`. Same weighting and sign-aligned quaternion
/// average the warp uses, evaluated over a subset.
#[allow(clippy::too_many_arguments)]
fn blend_excluding(
    skip_slot: usize,
    kept: &[usize],
    positions: &[Vec3],
    times: &[f64],
    rotations: &[Quat],
    translations: &[Vec3],
    point: Vec3,
    time: f64,
    sigma: f64,
    time_sigma: f64,
) -> (Quat, Vec3) {
    let inverse_two_sigma_squared = 1.0 / (2.0 * sigma * sigma);
    let inverse_two_time_sigma_squared = 1.0 / (2.0 * time_sigma * time_sigma);

    let mut nearest = kept[if skip_slot == 0 { 1 } else { 0 }];
    let mut nearest_weight = f64::NEG_INFINITY;
    let mut weight_sum = 0.0;
    let mut weighted: Vec<(usize, f64)> = Vec::with_capacity(kept.len());
    for (slot, &node) in kept.iter().enumerate() {
        if slot == skip_slot {
            continue;
        }
        let separation = spatial_distance(point, positions[node]);
        let dt = time - times[node];
        let exponent = separation * separation * inverse_two_sigma_squared
            + dt * dt * inverse_two_time_sigma_squared;
        let weight = (-exponent).exp();
        if weight > nearest_weight {
            nearest_weight = weight;
            nearest = node;
        }
        weighted.push((node, weight));
        weight_sum += weight;
    }

    if !(weight_sum > 1e-12) {
        return (rotations[nearest], translations[nearest]);
    }

    let reference = rotations[nearest];
    let mut sum = Quat::new(0.0, 0.0, 0.0, 0.0);
    let mut translation = Vec3::ZERO;
    for (node, raw_weight) in weighted {
        let weight = raw_weight / weight_sum;
        let rotation = rotations[node];
        let signed = if rotation_dot(rotation, reference) < 0.0 { -weight } else { weight };
        sum.x += signed * rotation.x;
        sum.y += signed * rotation.y;
        sum.z += signed * rotation.z;
        sum.w += signed * rotation.w;
        translation.x += weight * translations[node].x;
        translation.y += weight * translations[node].y;
        translation.z += weight * translations[node].z;
    }

    (sum.normalized_or_identity(), translation)
}

/// Resolve a Gaussian bandwidth for the current kept set: use `configured` when
/// positive, otherwise auto-pick 2× the mean nearest-neighbor spacing (matching
/// `graph::to_blend_arrays`), computed over just the kept coordinates.
fn resolved_sigma<T: Copy>(
    configured: f64,
    kept: &[usize],
    coordinates: &[T],
    metric: impl Fn(T, T) -> f64,
) -> f64 {
    if configured > 0.0 {
        return configured;
    }
    let count = kept.len();
    if count < 2 {
        return 1e-6;
    }
    let mut total = 0.0;
    for (slot, &node) in kept.iter().enumerate() {
        let mut nearest = f64::INFINITY;
        for (other_slot, &other) in kept.iter().enumerate() {
            if slot == other_slot {
                continue;
            }
            let distance = metric(coordinates[node], coordinates[other]);
            if distance < nearest {
                nearest = distance;
            }
        }
        total += nearest;
    }
    ((total / count as f64) * 2.0).max(1e-6)
}

fn spatial_distance(a: Vec3, b: Vec3) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn rotation_dot(a: Quat, b: Quat) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
}

fn apply(rotation: Quat, translation: Vec3, point: Vec3) -> Vec3 {
    let rotated = rotation.rotate(point);
    Vec3::new(rotated.x + translation.x, rotated.y + translation.y, rotated.z + translation.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::{generate_scene, LoopParams};
    use crate::warp::reform;

    #[test]
    fn target_node_count_is_respected() {
        let scene = generate_scene(&LoopParams::default());
        let thinned = sparsify(&scene.graph_delta, 12);
        assert_eq!(thinned.len(), 12);
        assert_eq!(thinned.transforms.len(), 12);
    }

    #[test]
    fn sparsify_is_a_passthrough_when_target_exceeds_node_count() {
        let scene = generate_scene(&LoopParams::default());
        let count = scene.graph_delta.len();
        let thinned = sparsify(&scene.graph_delta, count + 10);
        assert_eq!(thinned, scene.graph_delta);
    }

    /// A zero budget forbids any error, so every node is kept; a generous budget
    /// drops nodes. Kept ids stay a subset of the originals in both cases.
    #[test]
    fn error_budget_bounds_how_many_nodes_drop() {
        let scene = generate_scene(&LoopParams::default());
        let count = scene.graph_delta.len();

        let strict = sparsify_within_error(&scene.graph_delta, 0.0, None);
        assert_eq!(strict.len(), count, "zero budget must keep every node");

        let generous = sparsify_within_error(&scene.graph_delta, 1.0, None);
        assert!(generous.len() < count, "a generous budget should drop nodes");

        let original_ids: std::collections::HashSet<u64> =
            scene.graph_delta.nodes.iter().map(|node| node.id).collect();
        assert!(generous.nodes.iter().all(|node| original_ids.contains(&node.id)));
    }

    /// Warm-starting from a subset of ids begins the search there; the result is
    /// still a valid within-budget thinning (a subset of the originals).
    #[test]
    fn warm_start_seed_is_honored() {
        let scene = generate_scene(&LoopParams::default());
        let seed: Vec<u64> = scene.graph_delta.nodes.iter().map(|node| node.id).take(15).collect();
        let thinned = sparsify_within_error(&scene.graph_delta, 0.3, Some(&seed));
        let original_ids: std::collections::HashSet<u64> =
            scene.graph_delta.nodes.iter().map(|node| node.id).collect();
        assert!(!thinned.nodes.is_empty());
        assert!(thinned.nodes.iter().all(|node| original_ids.contains(&node.id)));
    }

    /// The whole point: dropping redundant nodes must still close the loop. Warp
    /// the drifted cloud with a heavily thinned correction and check the seam
    /// ends (earliest vs latest point) still overlap about as well as with the
    /// full graph.
    #[test]
    fn thinned_correction_still_closes_the_loop() {
        let scene = generate_scene(&LoopParams::default());
        let thinned = sparsify(&scene.graph_delta, 10);

        let full = reform(&scene.cloud, &scene.graph_delta);
        let sparse = reform(&scene.cloud, &thinned);

        let full_gap = seam_gap(&full, scene.cloud.timestamps.as_deref().unwrap());
        let sparse_gap = seam_gap(&sparse, scene.cloud.timestamps.as_deref().unwrap());

        // The thinned warp should not be dramatically worse than the full one.
        assert!(
            sparse_gap < full_gap + 2.0,
            "thinned seam gap {sparse_gap} vs full {full_gap}"
        );
    }

    fn seam_gap(cloud: &crate::point_cloud::PointCloud2, times: &[f64]) -> f64 {
        let mut earliest = 0usize;
        let mut latest = 0usize;
        for index in 1..times.len() {
            if times[index] < times[earliest] {
                earliest = index;
            }
            if times[index] > times[latest] {
                latest = index;
            }
        }
        let a = cloud.points[earliest];
        let b = cloud.points[latest];
        let dx = (a[0] - b[0]) as f64;
        let dy = (a[1] - b[1]) as f64;
        (dx * dx + dy * dy).sqrt()
    }
}
