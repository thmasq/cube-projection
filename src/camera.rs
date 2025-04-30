use glam::{Mat4, Vec3};

#[allow(dead_code)]
pub enum ProjectionType {
    Orthographic,
    Perspective,
}

pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,

    // Orthographic parameters
    pub left: f32,
    pub right: f32,
    pub bottom: f32,
    pub top: f32,

    // Common parameters
    pub near: f32,
    pub far: f32,

    // Perspective parameters
    pub fov: f32,
    pub aspect_ratio: f32,

    // Camera orientation
    pub yaw: f32,
    pub pitch: f32,

    // Projection type
    pub projection_type: ProjectionType,
}

impl Camera {
    pub fn new_perspective(
        position: Vec3,
        target: Vec3,
        up: Vec3,
        fov_degrees: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        // Calculate initial yaw and pitch from the given position and target
        let direction = (target - position).normalize();
        let pitch = (direction.y).asin();
        let yaw = direction.z.atan2(direction.x);

        Self {
            position,
            target,
            up,
            left: 0.0,   // Not used for perspective
            right: 0.0,  // Not used for perspective
            bottom: 0.0, // Not used for perspective
            top: 0.0,    // Not used for perspective
            near,
            far,
            fov: fov_degrees.to_radians(),
            aspect_ratio,
            yaw,
            pitch,
            projection_type: ProjectionType::Perspective,
        }
    }

    // Update the camera direction based on yaw and pitch
    fn update_camera_vectors(&mut self) {
        // Calculate new front vector
        let front = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize();

        // Update target based on position and front
        self.target = self.position + front;
    }

    // Movement methods for interactive camera control
    pub fn move_forward(&mut self, distance: f32) {
        let front = (self.target - self.position).normalize();
        self.position += front * distance;
        self.target += front * distance;
    }

    pub fn move_right(&mut self, distance: f32) {
        let front = (self.target - self.position).normalize();
        let right = front.cross(self.up).normalize();
        self.position += right * distance;
        self.target += right * distance;
    }

    pub fn move_up(&mut self, distance: f32) {
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        self.position += world_up * distance;
        self.target += world_up * distance;
    }

    // Rotate the camera with yaw and pitch (in radians)
    pub fn rotate(&mut self, yaw_delta: f32, pitch_delta: f32) {
        // Update yaw and pitch
        self.yaw += yaw_delta;
        self.pitch -= pitch_delta; // Inverted to match mouse movement

        // Constrain pitch to avoid gimbal lock
        self.pitch = self.pitch.clamp(-1.5, 1.5); // Roughly ±85 degrees

        // Update camera orientation
        self.update_camera_vectors();
    }

    // Zoom by changing the FOV (for perspective) or size (for orthographic)
    pub fn zoom(&mut self, amount: f32) {
        match self.projection_type {
            ProjectionType::Perspective => {
                // Change FOV for perspective camera
                self.fov = amount.mul_add(-0.05, self.fov).clamp(0.1, 2.0); // Limit FOV to sensible range
            }
            ProjectionType::Orthographic => {
                // Change size for orthographic camera
                let scale_factor = amount.mul_add(-0.05, 1.0);
                self.left *= scale_factor;
                self.right *= scale_factor;
                self.top *= scale_factor;
                self.bottom *= scale_factor;
            }
        }
    }

    // Update aspect ratio when window is resized
    pub const fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;
    }

    pub fn get_view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    pub fn get_projection_matrix(&self) -> Mat4 {
        match self.projection_type {
            ProjectionType::Orthographic => {
                // Create a flip Y matrix to invert the Y coordinates
                let flip_xy = Mat4::from_scale(Vec3::new(-1.0, -1.0, 1.0));

                // Combine the flip with the orthographic projection
                flip_xy
                    * Mat4::orthographic_rh(
                        self.left,
                        self.right,
                        self.bottom,
                        self.top,
                        self.near,
                        self.far,
                    )
            }
            ProjectionType::Perspective => {
                // For perspective, we'll use the standard perspective matrix
                Mat4::perspective_rh(self.fov, self.aspect_ratio, self.near, self.far)
            }
        }
    }

    pub fn get_view_proj_matrix(&self) -> Mat4 {
        self.get_projection_matrix() * self.get_view_matrix()
    }
}
