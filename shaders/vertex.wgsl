// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
}

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(vert: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Apply model and view-projection transformations
    let world_position = uniforms.model * vec4<f32>(vert.position, 1.0);
    out.clip_position = uniforms.view_proj * world_position;
    
    // Transform normal to world space
    let normal = (uniforms.model * vec4<f32>(vert.normal, 0.0)).xyz;
    out.normal = normalize(normal);
    
    // Use a flat white color instead of normal-based colors
    // This will result in untextured areas being flat white
    out.color = vec3<f32>(1.0, 1.0, 1.0);
    
    // Pass texture coordinates to fragment shader
    out.tex_coords = vec2<f32>(vert.tex_coords.x, 1.0 - vert.tex_coords.y);
    
    return out;
}
