use glam::{Mat4, Vec3};

pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub left: f32,
    pub right: f32,
    pub bottom: f32,
    pub top: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new_orthographic(position: Vec3, target: Vec3, up: Vec3, size: f32) -> Self {
        // For orthographic projection, we'll use size to determine the viewport dimensions
        let half_size = size / 2.0;

        // Calculate the distance from position to target for setting appropriate near/far planes
        let distance = (position - target).length();
        let near = distance * 0.5; // Place near plane halfway between camera and target
        let far = distance * 1.5; // Place far plane beyond the target

        Self {
            position,
            target,
            up,
            left: -half_size,
            right: half_size,
            bottom: -half_size,
            top: half_size,
            near,
            far,
        }
    }

    pub fn get_view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    pub fn get_projection_matrix(&self) -> Mat4 {
        // Create a flip Y matrix to invert the Y coordinates
        let flip_y = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0));

        // Combine the flip with the orthographic projection
        flip_y
            * Mat4::orthographic_rh(
                self.left,
                self.right,
                self.bottom,
                self.top,
                self.near,
                self.far,
            )
    }

    pub fn get_view_proj_matrix(&self) -> Mat4 {
        self.get_projection_matrix() * self.get_view_matrix()
    }
}

pub fn create_cube_cameras(min_bound: Vec3, max_bound: Vec3) -> [Camera; 6] {
    // Calculate the center of the bounding box
    let center = (min_bound + max_bound) * 0.5;

    // Calculate the dimensions of the bounding box
    let dimensions = max_bound - min_bound;

    // Calculate the orthographic size based on the largest dimension
    // with an additional 10% padding
    let max_dimension = dimensions.x.max(dimensions.y).max(dimensions.z);
    let ortho_size = max_dimension * 1.1;

    // Distance from center to camera - keep this relatively small for orthographic
    // Just enough to be outside the model
    let distance = max_dimension * 0.6;

    // Camera positions - at a distance from the center in each cardinal direction
    [
        // +X face (right)
        Camera::new_orthographic(
            center + Vec3::new(distance, 0.0, 0.0),
            center,
            Vec3::new(0.0, -1.0, 0.0), // Standard up vector
            ortho_size,
        ),
        // -X face (left)
        Camera::new_orthographic(
            center + Vec3::new(-distance, 0.0, 0.0),
            center,
            Vec3::new(0.0, -1.0, 0.0), // Standard up vector
            ortho_size,
        ),
        // +Y face (up)
        Camera::new_orthographic(
            center + Vec3::new(0.0, distance, 0.0),
            center,
            Vec3::new(0.0, 0.0, 1.0), // Standard up vector
            ortho_size,
        ),
        // -Y face (down)
        Camera::new_orthographic(
            center + Vec3::new(0.0, -distance, 0.0),
            center,
            Vec3::new(0.0, 0.0, -1.0), // Standard up vector
            ortho_size,
        ),
        // +Z face (front)
        Camera::new_orthographic(
            center + Vec3::new(0.0, 0.0, distance),
            center,
            Vec3::new(0.0, -1.0, 0.0), // Standard up vector
            ortho_size,
        ),
        // -Z face (back)
        Camera::new_orthographic(
            center + Vec3::new(0.0, 0.0, -distance),
            center,
            Vec3::new(0.0, -1.0, 0.0), // Standard up vector
            ortho_size,
        ),
    ]
}
