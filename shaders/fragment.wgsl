// Fragment shader

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply simple lighting model - diffuse only
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let diffuse = max(dot(in.normal, light_dir), 0.0);
    
    // Final color: mix of ambient and diffuse lighting
    let ambient = 0.2;
    let light_color = vec3<f32>(1.0, 1.0, 1.0);
    
    let base_color = in.color;
    let result = (ambient + diffuse) * base_color;
    
    return vec4<f32>(result, 1.0);
}
