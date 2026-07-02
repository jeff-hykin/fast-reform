//! PointCloud2 — the data structure carried through the warp unchanged except
//! for the XYZ positions. Kept deliberately close to the reference type: XYZ
//! plus optional per-point timestamps and intensities, and cloud-level metadata.

/// A point cloud. Positions are stored as f32 (like the reference), but the warp
/// computes in f64 for accuracy.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PointCloud2 {
    /// (M, 3) XYZ positions in world frame.
    pub points: Vec<[f32; 3]>,
    /// (M,) per-point observation timestamps in seconds. Binds each point to the
    /// pose-graph timeline. `None` means "unknown age".
    pub timestamps: Option<Vec<f64>>,
    /// (M,) per-point intensities, carried through unchanged.
    pub intensities: Option<Vec<f32>>,
    /// Coordinate frame name, carried through unchanged.
    pub frame_id: String,
    /// Cloud-level timestamp (seconds), carried through unchanged.
    pub ts: f64,
}

impl PointCloud2 {
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Build a cloud with the given positions and metadata copied from `self`.
    /// Used to produce the warped output while preserving everything else.
    pub fn with_points(&self, points: Vec<[f32; 3]>) -> PointCloud2 {
        PointCloud2 {
            points,
            timestamps: self.timestamps.clone(),
            intensities: self.intensities.clone(),
            frame_id: self.frame_id.clone(),
            ts: self.ts,
        }
    }
}
