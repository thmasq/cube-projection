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

        Self {
            position,
            target,
            up,
            left: -half_size,
            right: half_size,
            bottom: -half_size,
            top: half_size,
            near: 0.1,
            far: 100.0, // Placeholder, will be calculated based on mesh bounds
        }
    }

    // Calculate near and far planes based on mesh bounds and camera position
    pub fn update_depth_for_mesh(&mut self, min_bound: Vec3, max_bound: Vec3) {
        // Get view matrix to transform bounds to camera space
        let view_matrix = self.get_view_matrix();

        // Transform the bounding box corners to view space
        let corners = [
            min_bound,
            Vec3::new(min_bound.x, min_bound.y, max_bound.z),
            Vec3::new(min_bound.x, max_bound.y, min_bound.z),
            Vec3::new(min_bound.x, max_bound.y, max_bound.z),
            Vec3::new(max_bound.x, min_bound.y, min_bound.z),
            Vec3::new(max_bound.x, min_bound.y, max_bound.z),
            Vec3::new(max_bound.x, max_bound.y, min_bound.z),
            max_bound,
        ];

        // Find min/max z in view space
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;

        for corner in &corners {
            // Transform corner to view space (camera space)
            let corner_view = view_matrix.transform_point3(*corner);

            // In view space, camera looks down -Z, so we negate for proper depth comparison
            // (closer objects have smaller -Z values)
            let depth = -corner_view.z;

            min_z = min_z.min(depth);
            max_z = max_z.max(depth);
        }

        // Set near and far with a small buffer (5%)
        // Ensure near is always positive and at least 0.1 to avoid numerical issues
        let buffer = (max_z - min_z) * 0.05;
        self.near = (min_z - buffer).max(0.1);
        self.far = max_z + buffer;

        // Ensure far > near
        if self.far <= self.near {
            self.far = self.near + 1.0;
        }

        log::debug!(
            "Camera depth adjusted: near={}, far={}, range={}",
            self.near,
            self.far,
            self.far - self.near
        );
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

    // Position cameras at a reasonable distance - we'll adjust their
    // near/far planes based on the actual mesh later
    let distance = max_dimension * 0.8;

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
