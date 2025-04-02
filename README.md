# Cube Projection

A tool that projects 3D models onto the six faces of a cube, rendering the model from each direction with texture support.

## Building

```
cargo build --release
```

## Usage

```
./target/release/cube-projection -i </model/path.obj> -t </texture/path.png> -s <size of cube in pixels> -o <output directory>
```

## Arguments

- `-i, --input`: Path to input .obj file (required)
- `-t, --texture`: Path to texture image (optional)
- `-s, --size`: Size of output images in pixels (default: 512)
- `-o, --output-dir`: Directory to save output images (default: current directory)
- `-a, --aa-quality`: Anti-aliasing quality 0-3 (default: 2)
- `-b, --bg-color`: Background color as hex code (optional, default: gray)
  - Formats: RGB, RGBA, RRGGBB, or RRGGBBAA
  - Examples: `FF0000` (red), `00FF00FF` (opaque green), `00000000` (transparent)
