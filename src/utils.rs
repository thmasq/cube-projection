use anyhow::Result;

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
        r: f64::from(r) / 255.0,
        g: f64::from(g) / 255.0,
        b: f64::from(b) / 255.0,
        a: f64::from(a) / 255.0,
    })
}
