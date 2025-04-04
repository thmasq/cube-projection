use anyhow::Result;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

pub fn save_image(data: &[u8], width: u32, height: u32, path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);

    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder.write_header()?;
    writer.write_image_data(data)?;

    Ok(())
}

pub fn parse_hex_color(hex: &str) -> Result<wgpu::Color> {
    let hex = hex.trim_start_matches('#');

    let (r, g, b, a) = match hex.len() {
        3 => {
            // RGB format
            let r = u8::from_str_radix(&hex[0..1], 16)?;
            let g = u8::from_str_radix(&hex[1..2], 16)?;
            let b = u8::from_str_radix(&hex[2..3], 16)?;
            (r * 16 + r, g * 16 + g, b * 16 + b, 255)
        }
        4 => {
            // RGBA format
            let r = u8::from_str_radix(&hex[0..1], 16)?;
            let g = u8::from_str_radix(&hex[1..2], 16)?;
            let b = u8::from_str_radix(&hex[2..3], 16)?;
            let a = u8::from_str_radix(&hex[3..4], 16)?;
            (r * 16 + r, g * 16 + g, b * 16 + b, a * 16 + a)
        }
        6 => {
            // RRGGBB format
            let r = u8::from_str_radix(&hex[0..2], 16)?;
            let g = u8::from_str_radix(&hex[2..4], 16)?;
            let b = u8::from_str_radix(&hex[4..6], 16)?;
            (r, g, b, 255)
        }
        8 => {
            // RRGGBBAA format
            let r = u8::from_str_radix(&hex[0..2], 16)?;
            let g = u8::from_str_radix(&hex[2..4], 16)?;
            let b = u8::from_str_radix(&hex[4..6], 16)?;
            let a = u8::from_str_radix(&hex[6..8], 16)?;
            (r, g, b, a)
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Invalid hex color format. Use #RGB, #RGBA, #RRGGBB, or #RRGGBBAA"
            ));
        }
    };

    // Convert to 0.0-1.0 range
    Ok(wgpu::Color {
        r: r as f64 / 255.0,
        g: g as f64 / 255.0,
        b: b as f64 / 255.0,
        a: a as f64 / 255.0,
    })
}
