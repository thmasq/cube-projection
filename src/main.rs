mod camera;
mod mesh;
mod renderer;
mod utils;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

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
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

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

    // Create output directory if it doesn't exist
    if !args.output_dir.exists() {
        std::fs::create_dir_all(&args.output_dir)?;
    }

    // Load the mesh
    log::info!("Loading mesh from {:?}", args.input);
    let mesh = mesh::Mesh::load(&args.input)?;
    log::info!(
        "Loaded mesh with {} vertices and {} indices",
        mesh.vertices.len(),
        mesh.indices.len()
    );

    // Initialize renderer
    log::info!(
        "Initializing renderer with image size {}x{}",
        args.size,
        args.size
    );
    let mut renderer = pollster::block_on(renderer::Renderer::new(args.size, args.size))?;

    // Create the cameras for each cube face
    let cameras = camera::create_cube_cameras();

    // Render each cube face and save the images
    for (i, camera) in cameras.iter().enumerate() {
        let face_name = match i {
            0 => "positive_x",
            1 => "negative_x",
            2 => "positive_y",
            3 => "negative_y",
            4 => "positive_z",
            5 => "negative_z",
            _ => unreachable!(),
        };

        log::info!("Rendering {} face", face_name);
        let image_data = pollster::block_on(renderer.render(&mesh, camera))?;

        // Save the image
        let output_path = args.output_dir.join(format!("face_{}.png", face_name));
        log::info!("Saving image to {:?}", output_path);
        utils::save_image(&image_data, args.size, args.size, &output_path)?;
    }

    log::info!("Completed cube projection");
    Ok(())
}
