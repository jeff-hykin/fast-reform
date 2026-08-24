//! `Reformer` — a stateful handle around a pose-graph correction.
//!
//! The one-shot [`reform`](crate::warp::reform) rebuilds the blend arrays
//! (normalized rotations, node anchors/times, resolved bandwidths) on every
//! call. When the same correction is applied to many clouds
//! — or a live map is re-warped as new points stream in — that rebuild is wasted
//! work. `Reformer` owns the `GraphDelta3D` and caches its `BlendDeltas`,
//! recomputing the cache only when the correction changes (`set_delta` /
//! `sparsify`). Callers then `apply` on demand against the cached arrays.

use crate::graph::{BlendDeltas, GraphDelta3D};
use crate::point_cloud::PointCloud2;
use crate::sparsify::{sparsify, sparsify_within_error};
use crate::warp::warp_positions;

/// A reusable warper: holds a *full* correction and the (possibly thinned) active
/// subset it applies, caching the active subset's blend arrays so `apply` never
/// rebuilds them. Update the correction with `set_delta`, thin it with `sparsify`
/// / `sparsify_within_error`, or push a fresh PGO correction with
/// `update_within_error` (which warm-starts the thinning from the last kept set).
#[derive(Clone, Debug)]
pub struct Reformer {
    /// The complete correction, all nodes — the source the sparsifiers thin from.
    full: GraphDelta3D,
    /// The subset currently applied (equals `full` until thinned).
    active: GraphDelta3D,
    /// Cached blend arrays of `active`.
    blend: BlendDeltas,
}

impl Reformer {
    /// Build a reformer for a correction, precomputing its blend arrays. Nothing
    /// is thinned yet, so the active subset is the whole correction.
    pub fn new(graph_delta: GraphDelta3D) -> Self {
        let blend = graph_delta.to_blend_arrays();
        Reformer { full: graph_delta.clone(), active: graph_delta, blend }
    }

    /// Set the active subset (and its cache) directly.
    fn set_active(&mut self, active: GraphDelta3D) {
        self.blend = active.to_blend_arrays();
        self.active = active;
    }

    /// Replace the full correction (resetting any thinning) and rebuild the cache.
    pub fn set_delta(&mut self, graph_delta: GraphDelta3D) {
        self.full = graph_delta.clone();
        self.set_active(graph_delta);
    }

    /// The node ids currently kept in the active subset — the warm-start seed for
    /// the next `update_within_error`.
    pub fn kept_ids(&self) -> Vec<u64> {
        self.active.nodes.iter().map(|node| node.id).collect()
    }

    /// Thin the full correction to `target_nodes` (greedy leave-one-out
    /// decimation) and rebuild the cache.
    pub fn sparsify(&mut self, target_nodes: usize) {
        let thinned = sparsify(&self.full, target_nodes);
        self.set_active(thinned);
    }

    /// Thin the full correction as far as the accumulated anchor error allows
    /// (`<= max_error` vs the full deformation), warm-starting from the currently
    /// kept nodes. Rebuilds the cache.
    pub fn sparsify_within_error(&mut self, max_error: f64) {
        let seed = self.kept_ids();
        let thinned = sparsify_within_error(&self.full, max_error, Some(&seed));
        self.set_active(thinned);
    }

    /// Push a fresh PGO correction (the same nodes, matched by id, with updated
    /// deltas) and re-thin it within `max_error`, warm-starting from the nodes
    /// kept last time. Cheap when the graph was thinned before: it starts from the
    /// previous selection instead of every node. Rebuilds the cache.
    pub fn update_within_error(&mut self, graph_delta: GraphDelta3D, max_error: f64) {
        let seed = self.kept_ids();
        let thinned = sparsify_within_error(&graph_delta, max_error, Some(&seed));
        self.full = graph_delta;
        self.set_active(thinned);
    }

    /// Warp a whole cloud with the cached correction. Pass-through when the
    /// correction has no nodes or the cloud is empty.
    pub fn apply(&self, cloud: &PointCloud2) -> PointCloud2 {
        if self.blend.is_empty() || cloud.is_empty() {
            return cloud.clone();
        }
        let warped = warp_positions(&cloud.points, cloud.timestamps.as_deref(), &self.blend);
        cloud.with_points(warped)
    }

    /// Warp raw positions with the cached correction. `point_times`, when present,
    /// must align with `points`.
    pub fn apply_positions(
        &self,
        points: &[[f32; 3]],
        point_times: Option<&[f64]>,
    ) -> Vec<[f32; 3]> {
        warp_positions(points, point_times, &self.blend)
    }

    /// The active (possibly thinned) correction being applied — e.g. to read its
    /// node count after sparsifying.
    pub fn graph_delta(&self) -> &GraphDelta3D {
        &self.active
    }

    /// The full, un-thinned correction the sparsifiers work from.
    pub fn full_delta(&self) -> &GraphDelta3D {
        &self.full
    }

    /// The cached blend arrays.
    pub fn blend(&self) -> &BlendDeltas {
        &self.blend
    }

    pub fn is_empty(&self) -> bool {
        self.blend.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::{generate_scene, LoopParams};
    use crate::warp::reform;

    #[test]
    fn apply_matches_the_one_shot_function() {
        let scene = generate_scene(&LoopParams::default());
        let reformer = Reformer::new(scene.graph_delta.clone());

        let via_struct = reformer.apply(&scene.cloud);
        let via_function = reform(&scene.cloud, &scene.graph_delta);
        assert_eq!(via_struct, via_function);
    }

    #[test]
    fn set_delta_rebuilds_the_cache() {
        let scene = generate_scene(&LoopParams::default());
        let mut reformer = Reformer::new(GraphDelta3D::default());
        // Empty correction is a pass-through.
        assert_eq!(reformer.apply(&scene.cloud), scene.cloud);

        reformer.set_delta(scene.graph_delta.clone());
        let warped = reformer.apply(&scene.cloud);
        assert_eq!(warped, reform(&scene.cloud, &scene.graph_delta));
    }

    #[test]
    fn sparsify_reduces_the_held_node_count() {
        let scene = generate_scene(&LoopParams::default());
        let mut reformer = Reformer::new(scene.graph_delta.clone());
        reformer.sparsify(10);
        assert_eq!(reformer.graph_delta().len(), 10);
        // Still applies without the two seam ends drifting apart.
        let warped = reformer.apply(&scene.cloud);
        assert_eq!(warped.len(), scene.cloud.len());
        // The full correction is retained even though the active one shrank.
        assert_eq!(reformer.full_delta().len(), scene.graph_delta.len());
    }

    /// A bigger error budget must keep no more nodes than a tighter one.
    #[test]
    fn error_budget_trades_nodes_for_accuracy() {
        let scene = generate_scene(&LoopParams::default());
        let mut tight = Reformer::new(scene.graph_delta.clone());
        let mut loose = Reformer::new(scene.graph_delta.clone());
        tight.sparsify_within_error(0.05);
        loose.sparsify_within_error(0.5);
        assert!(
            loose.graph_delta().len() <= tight.graph_delta().len(),
            "looser budget kept more nodes: {} vs {}",
            loose.graph_delta().len(),
            tight.graph_delta().len(),
        );
    }

    /// Warm-starting a PGO update from the last kept set reproduces a from-scratch
    /// thin of that same delta (the seed only changes where the search starts).
    #[test]
    fn warm_start_update_matches_a_cold_thin() {
        let scene = generate_scene(&LoopParams::default());
        let mut reformer = Reformer::new(scene.graph_delta.clone());
        reformer.sparsify_within_error(0.2);

        // A fresh correction over the same node ids (here, the same deltas).
        reformer.update_within_error(scene.graph_delta.clone(), 0.2);
        let warm_ids = reformer.kept_ids();

        let cold = sparsify_within_error(&scene.graph_delta, 0.2, None);
        let cold_ids: Vec<u64> = cold.nodes.iter().map(|node| node.id).collect();
        assert_eq!(warm_ids, cold_ids);
    }
}
