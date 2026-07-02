//! `Reformer` — a stateful handle around a pose-graph correction.
//!
//! The one-shot [`apply_closure_to_cloud`](crate::warp::apply_closure_to_cloud)
//! rebuilds the blend arrays (normalized rotations, node anchors/times, resolved
//! bandwidths) on every call. When the same correction is applied to many clouds
//! — or a live map is re-warped as new points stream in — that rebuild is wasted
//! work. `Reformer` owns the `GraphDelta3D` and caches its `BlendDeltas`,
//! recomputing the cache only when the correction changes (`set_delta` /
//! `sparsify`). Callers then `apply` on demand against the cached arrays.

use crate::graph::{BlendDeltas, GraphDelta3D};
use crate::point_cloud::PointCloud2;
use crate::sparsify::sparsify;
use crate::warp::warp_positions;

/// A reusable warper: holds a correction and its precomputed blend arrays, and
/// applies it to clouds on demand. Update the correction with `set_delta` (or
/// thin it with `sparsify`) and the cache is rebuilt once, not per `apply`.
#[derive(Clone, Debug)]
pub struct Reformer {
    graph_delta: GraphDelta3D,
    blend: BlendDeltas,
}

impl Reformer {
    /// Build a reformer for a correction, precomputing its blend arrays.
    pub fn new(graph_delta: GraphDelta3D) -> Self {
        let blend = graph_delta.to_blend_arrays();
        Reformer { graph_delta, blend }
    }

    /// Replace the correction and rebuild the cached blend arrays.
    pub fn set_delta(&mut self, graph_delta: GraphDelta3D) {
        self.blend = graph_delta.to_blend_arrays();
        self.graph_delta = graph_delta;
    }

    /// Thin the held correction to `target_nodes` (greedy leave-one-out
    /// decimation) and rebuild the cache.
    pub fn sparsify(&mut self, target_nodes: usize) {
        let thinned = sparsify(&self.graph_delta, target_nodes);
        self.set_delta(thinned);
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

    /// The correction currently held (e.g. to read its node count after thinning).
    pub fn graph_delta(&self) -> &GraphDelta3D {
        &self.graph_delta
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
    use crate::warp::apply_closure_to_cloud;

    #[test]
    fn apply_matches_the_one_shot_function() {
        let scene = generate_scene(&LoopParams::default());
        let reformer = Reformer::new(scene.graph_delta.clone());

        let via_struct = reformer.apply(&scene.cloud);
        let via_function = apply_closure_to_cloud(&scene.cloud, &scene.graph_delta);
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
        assert_eq!(warped, apply_closure_to_cloud(&scene.cloud, &scene.graph_delta));
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
    }
}
