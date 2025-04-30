mod application;
mod camera;
mod input;
mod mesh;
mod renderer;
mod texture;
mod utils;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use winit::event_loop::EventLoop;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the input .obj file
    #[arg(short, long)]
    input: PathBuf,

    /// Size of the output images in pixels (both width and height)
    #[arg(short, long, default_value_t = 512)]
    size: u32,

    /// Output directory for the generated images
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,

    /// Anti-aliasing quality (0 = disabled, 1 = low, 2 = medium, 3 = high)
    #[arg(short = 'a', long, default_value_t = 2)]
    aa_quality: u8,

    /// Path to texture file (optional, but lack thereof will result in plain white model, with no lighting)
    #[arg(short = 't', long)]
    texture: Option<PathBuf>,

    /// Background color in hex format (RGB, RGBA, RRGGBB, or RRGGBBAA)
    /// Example: "FF0000" for red, "00FF00FF" for opaque green, "00000000" for transparent
    #[arg(short = 'b', long)]
    bg_color: Option<String>,

    /// Enable detailed performance metrics
    #[arg(short = 'm', long, default_value_t = false)]
    metrics: bool,

    /// Graphics backend to use (vulkan, opengl, metal, dx12, software, auto)
    /// Auto will try Vulkan first, then OpenGL/Metal, then software rendering
    #[arg(short = 'g', long, default_value = "auto")]
    graphics_backend: String,

    /// Run in interactive mode with a window instead of generating cube face images
    #[arg(short = 'w', long, default_value_t = false)]
    window: bool,
}

fn run_interactive_mode(args: &Args) -> Result<()> {
    // Initialize the logger
    env_logger::init();

    // Parse background color if provided
    let bg_color = if let Some(hex) = &args.bg_color {
        utils::parse_hex_color(hex).ok()
    } else {
        None
    };

    // Create event loop
    let event_loop = EventLoop::new()?;

    // Create and initialize application
    let mut app = application::Application::new();
    app.initialize(&event_loop, &args.input, args.texture.as_deref(), bg_color)?;

    // Run the event loop
    log::info!("Starting interactive mode");
    log::info!("Controls: WASD to move, Right mouse button + drag to rotate, Mouse wheel to zoom");
    log::info!("         Space to move up, Shift to move down, ESC to exit");

    event_loop.run_app(&mut app)?;

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    run_interactive_mode(&args)
}
