mod camera;
mod mesh;
mod renderer;
mod texture;
mod utils;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

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
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    // Start total time measurement
    let start_total = Instant::now();

    // Validate input file
    if !args.input.exists() {
        return Err(anyhow::anyhow!(
            "Input file does not exist: {:?}",
            args.input
        ));
    }
    if args.input.extension().unwrap_or_default() != "obj" {
        return Err(anyhow::anyhow!("Input must be an .obj file"));
    }

    if let Some(texture_path) = &args.texture {
        if !texture_path.exists() {
            return Err(anyhow::anyhow!(
                "Texture file does not exist: {:?}",
                texture_path
            ));
        }
        log::info!("Using texture from {}", texture_path.display());
    }

    // Create output directory if it doesn't exist
    if !args.output_dir.exists() {
        std::fs::create_dir_all(&args.output_dir)?;
    }

    // Load the mesh
    let mesh_start = Instant::now();
    log::info!("Loading mesh from {}", args.input.display());
    let mesh = mesh::Mesh::load(&args.input)?;
    if args.metrics {
        log::info!("Mesh loading: {:.2?}", mesh_start.elapsed());
    }

    // Calculate bounding box for proper camera positioning
    let bounds_start = Instant::now();
    let (min_bound, max_bound) = mesh.calculate_bounding_box();
    if args.metrics {
        log::info!("Bounding box calculation: {:.2?}", bounds_start.elapsed());
    }

    // Create the cameras for each cube face with appropriate distance
    let cameras_start = Instant::now();
    let mut cameras = camera::create_cube_cameras(min_bound, max_bound);
    // Update in parallel
    rayon::scope(|s| {
        for camera in cameras.iter_mut() {
            s.spawn(|_| {
                camera.update_depth_for_mesh(min_bound, max_bound);
            });
        }
    });
    if args.metrics {
        log::info!("Camera setup: {:.2?}", cameras_start.elapsed());
    }

    let renderer_start = Instant::now();
    let bg_color = if let Some(hex) = &args.bg_color {
        utils::parse_hex_color(hex).ok()
    } else {
        None
    };

    let renderer = pollster::block_on(renderer::Renderer::new(
        args.size,
        args.size,
        args.aa_quality,
        args.texture.as_deref(),
        bg_color,
    ))?;
    if args.metrics {
        log::info!("Renderer initialization: {:.2?}", renderer_start.elapsed());
    }

    // Render all faces using our optimized batch renderer
    let render_start = Instant::now();
    let face_images = pollster::block_on(renderer.render_cube_faces(&mesh, &cameras))?;
    if args.metrics {
        log::info!("Batch rendering: {:.2?}", render_start.elapsed());
    }

    // Save images in parallel
    let save_start = Instant::now();
    rayon::scope(|s| {
        for (i, image_data) in face_images.into_iter().enumerate() {
            let face_name = match i {
                0 => "positive_x",
                1 => "negative_x",
                2 => "positive_y",
                3 => "negative_y",
                4 => "positive_z",
                5 => "negative_z",
                _ => unreachable!(),
            };

            let output_path = args.output_dir.join(format!("face_{face_name}.png"));
            let size = args.size;

            s.spawn(move |_| {
                utils::save_image(&image_data, size, size, &output_path)
                    .expect("Failed to save image");
            });
        }
    });
    if args.metrics {
        log::info!("Image saving: {:.2?}", save_start.elapsed());
    }

    log::info!("Total execution time: {:.2?}", start_total.elapsed());
    Ok(())
}
