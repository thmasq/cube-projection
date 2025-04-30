use crate::camera::Camera;
use crate::input::InputState;
use crate::mesh::Mesh;
use crate::renderer::Renderer;
use anyhow::Result;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

pub struct Application {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    mesh: Option<Mesh>,
    camera: Option<Camera>,
    input_state: InputState,
    last_frame_time: Instant,
    frame_count: u64,
    frame_time_accumulator: Duration,
}

impl Application {
    pub fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            mesh: None,
            camera: None,
            input_state: InputState::default(),
            last_frame_time: Instant::now(),
            frame_count: 0,
            frame_time_accumulator: Duration::from_secs(0),
        }
    }

    pub fn initialize(
        &mut self,
        event_loop: &EventLoop<()>,
        mesh_path: &Path,
        texture_path: Option<&Path>,
        bg_color: Option<wgpu::Color>,
    ) -> Result<()> {
        let _ = bg_color;
        let _ = texture_path;
        let _ = event_loop;
        log::info!("Initializing application");

        // Load the mesh
        log::info!("Loading mesh from {}", mesh_path.display());
        let mesh = Mesh::load(mesh_path)?;

        // Calculate bounding box for proper camera positioning
        let (min_bound, max_bound) = mesh.calculate_bounding_box();

        // Create perspective camera
        let center = (min_bound + max_bound) * 0.5;
        let dimensions = max_bound - min_bound;
        let max_dimension = dimensions.x.max(dimensions.y).max(dimensions.z);
        let distance = max_dimension * 2.0;

        let camera = Camera::new_perspective(
            center + glam::Vec3::new(0.0, 0.0, distance),
            center,
            glam::Vec3::new(0.0, 1.0, 0.0),
            45.0,
            16.0 / 9.0, // Default aspect ratio, will be updated
            0.1,
            distance * 4.0,
        );

        self.mesh = Some(mesh);
        self.camera = Some(camera);

        log::info!("Application initialization complete");

        Ok(())
    }

    fn update(&mut self, delta_time: f32) {
        if let Some(camera) = &mut self.camera {
            // Update camera based on input
            if self.input_state.keyboard.key_pressed(KeyCode::KeyW) {
                camera.move_forward(delta_time * 2.0);
            }
            if self.input_state.keyboard.key_pressed(KeyCode::KeyS) {
                camera.move_forward(-delta_time * 2.0);
            }
            if self.input_state.keyboard.key_pressed(KeyCode::KeyA) {
                camera.move_right(-delta_time * 2.0);
            }
            if self.input_state.keyboard.key_pressed(KeyCode::KeyD) {
                camera.move_right(delta_time * 2.0);
            }
            if self.input_state.keyboard.key_pressed(KeyCode::Space) {
                camera.move_up(delta_time * 2.0);
            }
            if self.input_state.keyboard.key_pressed(KeyCode::ShiftLeft) {
                camera.move_up(-delta_time * 2.0);
            }

            // Apply mouse rotation if right button is pressed
            if self.input_state.mouse.button_pressed(MouseButton::Right) {
                let (dx, dy) = self.input_state.mouse.movement();
                if dx != 0.0 || dy != 0.0 {
                    camera.rotate(dx * 0.005, dy * 0.005);
                }
            }

            // Reset mouse movement after applying
            self.input_state.mouse.reset_movement();
        }
    }

    fn render(&mut self) -> Result<()> {
        if let (Some(_window), Some(renderer), Some(mesh), Some(camera)) = (
            &self.window.as_ref(),
            &mut self.renderer,
            &self.mesh,
            &self.camera,
        ) {
            renderer.render_to_window(mesh, camera)?;

            // Calculate and log FPS every second
            self.frame_count += 1;
            let now = Instant::now();
            let frame_time = now.duration_since(self.last_frame_time);
            self.last_frame_time = now;

            self.frame_time_accumulator += frame_time;
            if self.frame_time_accumulator >= Duration::from_secs(1) {
                let fps = self.frame_count as f64 / self.frame_time_accumulator.as_secs_f64();
                log::info!("FPS: {fps:.1}");
                self.frame_count = 0;
                self.frame_time_accumulator = Duration::from_secs(0);
            }
        }

        Ok(())
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = WindowAttributes::default().with_title("3D Model Viewer");
        match event_loop.create_window(window_attributes) {
            Ok(window) => {
                // Get window size for aspect ratio calculation
                let window_size = window.inner_size();
                let aspect_ratio = window_size.width as f32 / window_size.height as f32;

                // Update camera aspect ratio
                if let Some(camera) = &mut self.camera {
                    camera.set_aspect_ratio(aspect_ratio);
                }

                // Store window
                self.window = Some(window.into());

                // Now create renderer with window reference
                // We can safely get a reference because we just stored it
                let window_ref = self.window.as_ref().unwrap();

                let renderer =
                    pollster::block_on(Renderer::new_with_window(window_ref, None, None))
                        .expect("Failed to create renderer");

                self.renderer = Some(renderer);
            }
            Err(err) => {
                eprintln!("Error creating window: {err}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested; exiting application");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                log::info!("Window resized to {}x{}", size.width, size.height);

                // Update camera aspect ratio
                if let Some(camera) = &mut self.camera {
                    camera.set_aspect_ratio(size.width as f32 / size.height as f32);
                }

                // Resize renderer surface
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }

                // Request redraw
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                // Update your state first
                let now = Instant::now();
                let delta_time = if self.last_frame_time == Instant::now() {
                    1.0 / 60.0 // Default to 60 FPS for first frame
                } else {
                    now.duration_since(self.last_frame_time).as_secs_f32()
                };

                self.update(delta_time);

                // 1) borrow window immutably, then end that borrow immediately
                if let Some(window) = &self.window {
                    window.pre_present_notify();
                } // ← borrow of `&self.window` ends here

                // 2) now that no immutable borrow is active, call render()
                if let Err(err) = self.render() {
                    log::error!("Render error: {err}");
                }

                // 3) borrow window again (fresh immutable borrow) to request redraw
                if let Some(window) = &self.window {
                    window.request_redraw();
                } // ← and this borrow ends here
            }

            WindowEvent::KeyboardInput { event, .. } => {
                self.input_state.keyboard.process_keyboard_event(&event);

                // Handle ESC key to exit
                if event.physical_key == PhysicalKey::Code(KeyCode::Escape) && event.state == ElementState::Pressed {
                    log::info!("Escape key pressed; exiting application");
                    event_loop.exit();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input_state
                    .mouse
                    .process_cursor_moved(position.x, position.y);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.input_state.mouse.process_mouse_input(button, state);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.input_state.mouse.process_mouse_wheel(delta);

                // Zoom camera with mouse wheel
                if let Some(camera) = &mut self.camera {
                    let zoom_amount = self.input_state.mouse.wheel_delta() * 0.1;
                    if zoom_amount != 0.0 {
                        camera.zoom(zoom_amount);
                    }
                }

                // Reset wheel delta after applying
                self.input_state.mouse.reset_wheel_delta();
            }
            _ => {}
        }
    }
}
