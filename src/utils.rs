use anyhow::Result;
use std::path::Path;

pub fn save_image(data: &[u8], width: u32, height: u32, path: &Path) -> Result<()> {
    let buffer = image::RgbaImage::from_raw(width, height, data.to_vec())
        .ok_or_else(|| anyhow::anyhow!("Failed to create image from raw data"))?;

    buffer.save(path)?;
    Ok(())
}
