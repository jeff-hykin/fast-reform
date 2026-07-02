//! The pose-graph correction (GraphDelta3D) plus the sorted, ready-to-blend form
//! the warp consumes. A GraphDelta3D pairs each keyframe node (with its
//! timestamp) with the SE(3) delta PGO applied to it (`post = delta * pre`).

use crate::math::{lerp_vec3, nlerp, Quat, Vec3};

/// The SE(3) correction applied to one keyframe. Left-multiplied world-frame delta.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
}

impl Transform {
    pub const IDENTITY: Transform = Transform {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };

    /// This delta scaled between identity (`amount = 0`) and its full value
    /// (`amount = 1`). Used by the demo to sweep a scene from open loop to
    /// closed loop as a slider moves.
    pub fn scaled_from_identity(self, amount: f64) -> Transform {
        Transform {
            translation: lerp_vec3(Vec3::ZERO, self.translation, amount),
            rotation: nlerp(Quat::IDENTITY, self.rotation, amount),
        }
    }
}

/// One keyframe node. `position` is the keyframe's spatial anchor in the cloud's
/// (pre-correction / open-loop) frame — the warp blends node deltas by how close
/// a point is to these anchors, so the correction is a smooth function of *space*.
/// `ts` is kept for context but no longer drives the warp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    pub ts: f64,
    pub position: Vec3,
}

/// A pose-graph correction: aligned `nodes` and `transforms` of equal length.
///
/// A point's correction is a proximity-weighted average of the node deltas with
/// weight `exp(-d^2 / (2 sigma^2)) * exp(-dt^2 / (2 time_sigma^2))`, where `d` is
/// spatial distance to the node anchor and `dt` is the gap between the point's
/// observation time and the node's time.
///
/// - The **spatial** term makes the field smooth in space, so neighboring points
///   get near-identical corrections and the warp stays locally space-preserving
///   (no fold-over).
/// - The **time** term disambiguates a loop *seam*: where the drifted trajectory
///   crosses back over itself, two segments share the same space but need opposite
///   corrections. Spatial distance alone would average them (closing nothing);
///   the time term keeps a point bound to the segment it was actually observed on.
///
/// `blend_sigma` (space) and `blend_time_sigma` (time) are the Gaussian
/// bandwidths; `0` means auto-pick from node spacing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphDelta3D {
    pub nodes: Vec<Node>,
    pub transforms: Vec<Transform>,
    pub blend_sigma: f64,
    pub blend_time_sigma: f64,
}

impl GraphDelta3D {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// This correction with every delta scaled toward identity by `amount`
    /// (0 = no correction / open loop, 1 = full correction / closed loop).
    pub fn scaled(&self, amount: f64) -> GraphDelta3D {
        GraphDelta3D {
            nodes: self.nodes.clone(),
            transforms: self
                .transforms
                .iter()
                .map(|transform| transform.scaled_from_identity(amount))
                .collect(),
            blend_sigma: self.blend_sigma,
            blend_time_sigma: self.blend_time_sigma,
        }
    }

    /// Normalize quaternions and gather the node anchors + times + deltas the warp
    /// weights over. Resolves both bandwidths (auto-picks from spacing when `<= 0`).
    pub fn to_blend_arrays(&self) -> BlendDeltas {
        let positions: Vec<Vec3> = self.nodes.iter().map(|node| node.position).collect();
        let times: Vec<f64> = self.nodes.iter().map(|node| node.ts).collect();
        let rotations = self
            .transforms
            .iter()
            .map(|transform| transform.rotation.normalized_or_identity())
            .collect();
        let translations = self.transforms.iter().map(|transform| transform.translation).collect();

        let sigma = if self.blend_sigma > 0.0 {
            self.blend_sigma
        } else {
            auto_sigma(mean_nearest(&positions, |a, b| distance(a, b)))
        };
        let time_sigma = if self.blend_time_sigma > 0.0 {
            self.blend_time_sigma
        } else {
            let time_points: Vec<Vec3> = times.iter().map(|&t| Vec3::new(t, 0.0, 0.0)).collect();
            auto_sigma(mean_nearest(&time_points, |a, b| (a.x - b.x).abs()))
        };

        BlendDeltas { positions, times, rotations, translations, sigma, time_sigma }
    }
}

fn distance(a: Vec3, b: Vec3) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Mean nearest-neighbor spacing under a caller-supplied metric.
fn mean_nearest(points: &[Vec3], metric: impl Fn(Vec3, Vec3) -> f64) -> f64 {
    let count = points.len();
    if count < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..count {
        let mut nearest = f64::INFINITY;
        for j in 0..count {
            if i == j {
                continue;
            }
            let d = metric(points[i], points[j]);
            if d < nearest {
                nearest = d;
            }
        }
        total += nearest;
    }
    total / count as f64
}

/// Widen a mean spacing into a Gaussian bandwidth: ~2 spacings overlaps several
/// neighbors so the blend stays smooth (non-folding).
fn auto_sigma(mean_spacing: f64) -> f64 {
    (mean_spacing * 2.0).max(1e-6)
}

/// Node anchor positions + times paired with their (normalized) deltas, for the
/// combined space+time proximity blend.
#[derive(Clone, Debug)]
pub struct BlendDeltas {
    pub positions: Vec<Vec3>,
    pub times: Vec<f64>,
    pub rotations: Vec<Quat>,
    pub translations: Vec<Vec3>,
    pub sigma: f64,
    pub time_sigma: f64,
}

impl BlendDeltas {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}
