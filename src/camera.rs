use glam::{Mat4, Vec3};

pub struct Camera {
    position: Vec3,
    target: Vec3,
    up: Vec3,
    aspect: f32,
    fov_y: f32,
    z_near: f32,
    z_far: f32,
}

impl Camera {
    pub fn new(position: Vec3, target: Vec3, up: Vec3) -> Self {
        Self {
            position,
            target,
            up,
            aspect: 1.0,                  // Square aspect ratio for cube faces
            fov_y: 90.0_f32.to_radians(), // 90 degrees for cube map
            z_near: 0.1,
            z_far: 100.0,
        }
    }

    pub fn get_view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    pub fn get_projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.z_near, self.z_far)
    }

    pub fn get_view_proj_matrix(&self) -> Mat4 {
        self.get_projection_matrix() * self.get_view_matrix()
    }
}

pub fn create_cube_cameras() -> [Camera; 6] {
    // The cameras are positioned at the origin and look in the direction of each face
    // of a cube. The up vector is chosen to ensure proper orientation.
    [
        // +X face (right)
        Camera::new(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        ),
        // -X face (left)
        Camera::new(
            Vec3::ZERO,
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        ),
        // +Y face (up)
        Camera::new(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ),
        // -Y face (down)
        Camera::new(
            Vec3::ZERO,
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        ),
        // +Z face (front)
        Camera::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, -1.0, 0.0),
        ),
        // -Z face (back)
        Camera::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, -1.0, 0.0),
        ),
    ]
}
