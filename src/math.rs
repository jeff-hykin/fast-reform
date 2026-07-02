//! Small SE(3) math: vectors, quaternions, and the blend primitives (nlerp for
//! rotation, lerp for translation). Blend parameters are allowed outside [0, 1]
//! so callers can extrapolate past the ends of the node range.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }
}

/// Linear interpolation/extrapolation between two translations. `amount`
/// outside [0, 1] extrapolates along the same line.
pub fn lerp_vec3(start: Vec3, end: Vec3, amount: f64) -> Vec3 {
    Vec3 {
        x: start.x + (end.x - start.x) * amount,
        y: start.y + (end.y - start.y) * amount,
        z: start.z + (end.z - start.z) * amount,
    }
}

/// Quaternion in (x, y, z, w) order — matches the reference implementation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Quat {
    pub const IDENTITY: Quat = Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Quat { x, y, z, w }
    }

    /// Normalize, falling back to identity for a zero-norm quaternion (matches
    /// the reference's zero-quaternion handling).
    pub fn normalized_or_identity(self) -> Quat {
        let norm = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        if norm == 0.0 {
            Quat::IDENTITY
        } else {
            let inverse = 1.0 / norm;
            Quat {
                x: self.x * inverse,
                y: self.y * inverse,
                z: self.z * inverse,
                w: self.w * inverse,
            }
        }
    }

    fn dot(self, other: Quat) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    fn negated(self) -> Quat {
        Quat { x: -self.x, y: -self.y, z: -self.z, w: -self.w }
    }

    /// Inverse rotation for a unit quaternion.
    pub fn conjugate(self) -> Quat {
        Quat { x: -self.x, y: -self.y, z: -self.z, w: self.w }
    }

    /// Rotate a vector by this (assumed unit) quaternion: q * v * q^-1.
    pub fn rotate(self, vector: Vec3) -> Vec3 {
        // t = 2 * cross(q_xyz, v); result = v + q_w * t + cross(q_xyz, t)
        let cross_x = self.y * vector.z - self.z * vector.y;
        let cross_y = self.z * vector.x - self.x * vector.z;
        let cross_z = self.x * vector.y - self.y * vector.x;

        let t_x = 2.0 * cross_x;
        let t_y = 2.0 * cross_y;
        let t_z = 2.0 * cross_z;

        let cross_t_x = self.y * t_z - self.z * t_y;
        let cross_t_y = self.z * t_x - self.x * t_z;
        let cross_t_z = self.x * t_y - self.y * t_x;

        Vec3 {
            x: vector.x + self.w * t_x + cross_t_x,
            y: vector.y + self.w * t_y + cross_t_y,
            z: vector.z + self.w * t_z + cross_t_z,
        }
    }
}

/// Normalized linear interpolation between two rotations. Cheaper than slerp;
/// the double-cover sign is resolved so the short path is taken. `amount`
/// outside [0, 1] extrapolates (still renormalized to a unit quaternion).
pub fn nlerp(start: Quat, end: Quat, amount: f64) -> Quat {
    let aligned_end = if start.dot(end) < 0.0 { end.negated() } else { end };
    let blended = Quat {
        x: start.x + (aligned_end.x - start.x) * amount,
        y: start.y + (aligned_end.y - start.y) * amount,
        z: start.z + (aligned_end.z - start.z) * amount,
        w: start.w + (aligned_end.w - start.w) * amount,
    };
    blended.normalized_or_identity()
}
