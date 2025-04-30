use std::collections::HashMap;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta};
use winit::keyboard::{KeyCode, PhysicalKey};

#[derive(Default)]
pub struct InputState {
    pub keyboard: KeyboardState,
    pub mouse: MouseState,
}

#[derive(Default)]
pub struct KeyboardState {
    keys: HashMap<KeyCode, bool>,
}


impl KeyboardState {
    pub fn process_keyboard_event(&mut self, event: &KeyEvent) {
        if let PhysicalKey::Code(key_code) = event.physical_key {
            let is_pressed = event.state == ElementState::Pressed;
            self.keys.insert(key_code, is_pressed);
        }
    }

    pub fn key_pressed(&self, key_code: KeyCode) -> bool {
        *self.keys.get(&key_code).unwrap_or(&false)
    }
}

pub struct MouseState {
    buttons: HashMap<MouseButton, bool>,
    cursor_x: f64,
    cursor_y: f64,
    prev_cursor_x: f64,
    prev_cursor_y: f64,
    wheel_delta: f32,
}

impl Default for MouseState {
    fn default() -> Self {
        Self {
            buttons: HashMap::new(),
            cursor_x: 0.0,
            cursor_y: 0.0,
            prev_cursor_x: 0.0,
            prev_cursor_y: 0.0,
            wheel_delta: 0.0,
        }
    }
}

impl MouseState {
    pub const fn process_cursor_moved(&mut self, x: f64, y: f64) {
        self.prev_cursor_x = self.cursor_x;
        self.prev_cursor_y = self.cursor_y;
        self.cursor_x = x;
        self.cursor_y = y;
    }

    pub fn process_mouse_input(&mut self, button: MouseButton, state: ElementState) {
        let is_pressed = state == ElementState::Pressed;
        self.buttons.insert(button, is_pressed);
    }

    pub fn process_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        match delta {
            MouseScrollDelta::LineDelta(_, y) => {
                self.wheel_delta += y;
            }
            MouseScrollDelta::PixelDelta(pos) => {
                // Convert pixel delta to a reasonable value
                self.wheel_delta += (pos.y / 100.0) as f32;
            }
        }
    }

    pub fn button_pressed(&self, button: MouseButton) -> bool {
        *self.buttons.get(&button).unwrap_or(&false)
    }

    pub fn movement(&self) -> (f32, f32) {
        (
            (self.cursor_x - self.prev_cursor_x) as f32,
            (self.cursor_y - self.prev_cursor_y) as f32,
        )
    }

    pub const fn reset_movement(&mut self) {
        self.prev_cursor_x = self.cursor_x;
        self.prev_cursor_y = self.cursor_y;
    }

    pub const fn wheel_delta(&self) -> f32 {
        self.wheel_delta
    }

    pub const fn reset_wheel_delta(&mut self) {
        self.wheel_delta = 0.0;
    }
}
