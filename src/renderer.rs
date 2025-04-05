use crate::camera::Camera;
use crate::mesh::{Mesh, Vertex};
use crate::texture::Texture;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use std::path::Path;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
}

#[allow(dead_code)]
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    sample_count: u32,
    background_color: wgpu::Color,
    ping_buffers: Vec<wgpu::Buffer>,
    pong_buffers: Vec<wgpu::Buffer>,
    current_frame: usize,
    buffer_alignment: u64,
    aligned_bytes_per_row: u32,
}

impl Renderer {
    pub async fn new(
        width: u32,
        height: u32,
        aa_quality: u8,
        texture_path: Option<&Path>,
        background_color: Option<wgpu::Color>,
        backend_preference: &str,
    ) -> Result<Self> {
        // Select appropriate backend based on preference
        let backends = match backend_preference.to_lowercase().as_str() {
            "vulkan" => wgpu::Backends::VULKAN,
            "opengl" => wgpu::Backends::GL,
            "metal" => wgpu::Backends::METAL,
            "dx12" => wgpu::Backends::DX12,
            "software" => {
                // For software rendering, set environment variable if not already set
                if std::env::var("WGPU_ADAPTER_NAME").is_err() {
                    unsafe { std::env::set_var("WGPU_ADAPTER_NAME", "llvmpipe") };
                }
                wgpu::Backends::GL
            }
            "auto" => {
                // Auto-selection with fallback priority
                if cfg!(target_os = "macos") {
                    // On macOS, prioritize Metal over OpenGL
                    wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::GL
                } else {
                    // On other platforms
                    wgpu::Backends::VULKAN | wgpu::Backends::GL | wgpu::Backends::DX12
                }
            }
            _ => {
                log::warn!(
                    "Unknown backend '{}', falling back to auto-selection",
                    backend_preference
                );
                wgpu::Backends::all()
            }
        };

        // Initialize WGPU with selected backends
        log::info!(
            "Initializing graphics with backend preference: {}",
            backend_preference
        );
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter_result = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await;

        // Try to get an adapter with the preferred backend
        let adapter = match (backend_preference.to_lowercase().as_str(), adapter_result) {
            ("vulkan", None) => {
                return Err(anyhow::anyhow!(
                    "Vulkan backend requested but not available. Try '-g opengl' or '-g auto'"
                ));
            }
            ("opengl", None) => {
                return Err(anyhow::anyhow!(
                    "OpenGL backend requested but not available. Try '-g vulkan' or '-g auto'"
                ));
            }
            ("metal", None) => {
                return Err(anyhow::anyhow!(
                    "Metal backend requested but not available. Try '-g opengl' or '-g auto'"
                ));
            }
            ("dx12", None) => {
                return Err(anyhow::anyhow!(
                    "DirectX 12 backend requested but not available. Try '-g vulkan' or '-g auto'"
                ));
            }
            ("software", None) => {
                return Err(anyhow::anyhow!(
                    "Software rendering requested but failed. Check your system configuration."
                ));
            }
            (_, None) => {
                return Err(anyhow::anyhow!(
                    "Failed to find a suitable GPU adapter with any backend"
                ));
            }
            (_, Some(adapter)) => adapter,
        };

        // Log which backend was actually chosen
        log::info!(
            "Using adapter: {} (backend: {:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;

        // Determine sample count based on quality and adapter capabilities
        let sample_flags = adapter
            .get_texture_format_features(wgpu::TextureFormat::Rgba8Unorm)
            .flags;

        // Map quality level to sample count
        let requested_samples = match aa_quality {
            0 => 1, // No AA
            1 => 2, // Low
            2 => 4, // Medium (default)
            _ => 8, // High or any other value
        };

        // Check adapter support
        let max_sample_count =
            if sample_flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X8) {
                8
            } else if sample_flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4) {
                4
            } else if sample_flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X2) {
                2
            } else {
                1 // No MSAA support
            };

        // Use requested sample count or the maximum available if lower
        let sample_count = requested_samples.min(max_sample_count);

        if sample_count > 1 {
            log::info!("Using MSAA with {} samples", sample_count);
        } else {
            log::info!("MSAA disabled");
        }

        // Create bind group layout for uniforms
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Uniform Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Create bind group layout for texture
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Texture Bind Group Layout"),
                entries: &[
                    // Texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Load texture or create default white texture
        let texture = match texture_path {
            Some(path) => Texture::load(&device, &queue, path)?,
            None => {
                log::info!("No texture provided. Using default white texture.");
                Texture::create_default(&device, &queue)
            }
        };

        // Create the texture bind group
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ],
        });

        // Create vertex shader module
        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/vertex.wgsl"
            ))),
        });

        // Create fragment shader module
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/fragment.wgsl"
            ))),
        });

        // Create render pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let background_color = background_color.unwrap_or(wgpu::Color {
            r: 0.5,
            g: 0.5,
            b: 0.5,
            a: 1.0,
        });

        // Calculate optimal buffer alignment
        let limits = device.limits();
        let buffer_alignment = limits.min_storage_buffer_offset_alignment as u64;

        // Calculate aligned bytes per row
        let bytes_per_row = width * 4;
        let aligned_bytes_per_row = ((bytes_per_row + 255) / 256) * 256;

        Ok(Self {
            device,
            queue,
            pipeline,
            uniform_bind_group_layout,
            texture_bind_group_layout,
            texture_bind_group,
            width,
            height,
            sample_count,
            background_color,
            ping_buffers: Vec::new(),
            pong_buffers: Vec::new(),
            current_frame: 0,
            buffer_alignment,
            aligned_bytes_per_row,
        })
    }

    pub fn init_readback_buffers(&mut self, face_count: usize) {
        let output_buffer_size = self.height as u64 * self.aligned_bytes_per_row as u64;

        // Create ping-pong buffers for each face
        for i in 0..face_count {
            // Ping buffer
            let ping = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Ping Buffer {}", i)),
                size: output_buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Pong buffer
            let pong = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Pong Buffer {}", i)),
                size: output_buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            self.ping_buffers.push(ping);
            self.pong_buffers.push(pong);
        }
    }

    pub async fn render_cube_faces(
        &mut self,
        mesh: &Mesh,
        cameras: &[Camera; 6],
    ) -> Result<Vec<Vec<u8>>> {
        // Ensure we have ping-pong buffers
        if self.ping_buffers.is_empty() {
            self.init_readback_buffers(6);
        }

        // Prepare GPU resources once for all renders
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        // Get current set of buffers to write to
        let current_buffers = if self.current_frame % 2 == 0 {
            &self.ping_buffers
        } else {
            &self.pong_buffers
        };

        // Prepare texture extent once for all faces
        let texture_extent = wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        };

        // Create textures for all faces
        let (render_textures, msaa_textures, depth_textures) = self.create_face_textures(6);

        // Create a single CommandEncoder for all rendering
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batch Render Encoder"),
            });

        // Submit all render passes first
        for (i, camera) in cameras.iter().enumerate() {
            let uniform_bind_group = self.create_uniform_bind_group(camera);

            let texture_view =
                render_textures[i].create_view(&wgpu::TextureViewDescriptor::default());
            let msaa_view = msaa_textures
                .get(i)
                .map(|tex| tex.create_view(&wgpu::TextureViewDescriptor::default()));
            let depth_view = depth_textures[i].create_view(&wgpu::TextureViewDescriptor::default());

            // Record render pass
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(&format!("Render Pass {}", i)),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: msaa_view.as_ref().unwrap_or(&texture_view),
                        resolve_target: if self.sample_count > 1 {
                            Some(&texture_view)
                        } else {
                            None
                        },
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.background_color),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });

                render_pass.set_pipeline(&self.pipeline);
                render_pass.set_bind_group(0, &uniform_bind_group, &[]);
                render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..u32::try_from(mesh.indices.len()).unwrap(), 0, 0..1);
            }

            // Record copy texture to buffer with optimal alignment
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &render_textures[i],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &current_buffers[i],
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(self.aligned_bytes_per_row),
                        rows_per_image: Some(self.height),
                    },
                },
                texture_extent,
            );
        }

        // Submit all commands at once to allow the GPU to batch operations
        self.queue.submit(std::iter::once(encoder.finish()));

        // Create shared buffer to collect results
        let results = Arc::new(Mutex::new(vec![Vec::new(); 6]));

        // Start mapping all buffers in parallel
        let mut mapping_receivers = Vec::with_capacity(current_buffers.len());

        for (i, buffer) in current_buffers.iter().enumerate() {
            let buffer_slice = buffer.slice(..);

            // Create a channel to receive the mapping result
            let (tx, rx) = std::sync::mpsc::channel();

            // Map the buffer with a callback
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });

            mapping_receivers.push((i, buffer_slice, rx));
        }

        // Wait for GPU work to complete
        self.device.poll(wgpu::Maintain::Wait);

        // Wait for all mapping operations to complete
        for (_, _, rx) in &mapping_receivers {
            rx.recv().unwrap().expect("Failed to map buffer");
        }

        // Process mapped buffers in parallel using rayon::scope
        // This ensures the threads don't outlive the scope of this method
        rayon::scope(|s| {
            for (i, buffer_slice, _) in mapping_receivers {
                let results_clone = Arc::clone(&results);
                let width = self.width;
                let height = self.height;
                let aligned_bytes_per_row = self.aligned_bytes_per_row;

                s.spawn(move |_| {
                    // Get the buffer data
                    let data = buffer_slice.get_mapped_range();

                    // Copy only the relevant bytes, skipping padding
                    let mut face_data = Vec::with_capacity((width * height * 4) as usize);

                    for row in 0..height {
                        let row_start = (row * aligned_bytes_per_row) as usize;
                        let row_end = row_start + (width * 4) as usize;
                        face_data.extend_from_slice(&data[row_start..row_end]);
                    }

                    // Release the mapped data
                    drop(data);

                    // Store the result
                    let mut results = results_clone.lock().unwrap();
                    results[i] = face_data;
                });
            }
        });

        // Unmap all buffers after processing
        for buffer in current_buffers {
            buffer.unmap();
        }

        // Toggle the current frame for ping-pong buffering
        self.current_frame += 1;

        // Return the results
        Ok(Arc::try_unwrap(results).unwrap().into_inner().unwrap())
    }

    // Create textures for multiple faces at once
    fn create_face_textures(
        &self,
        count: usize,
    ) -> (Vec<wgpu::Texture>, Vec<wgpu::Texture>, Vec<wgpu::Texture>) {
        let texture_extent = wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        };

        let render_textures: Vec<_> = (0..count)
            .map(|i| {
                self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("Render Texture {}", i)),
                    size: texture_extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
            })
            .collect();

        let msaa_textures: Vec<_> = if self.sample_count > 1 {
            (0..count)
                .map(|i| {
                    self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some(&format!("Multisampled Texture {}", i)),
                        size: texture_extent,
                        mip_level_count: 1,
                        sample_count: self.sample_count,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        view_formats: &[],
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        let depth_textures: Vec<_> = (0..count)
            .map(|i| {
                self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("Depth Texture {}", i)),
                    size: texture_extent,
                    mip_level_count: 1,
                    sample_count: self.sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
            })
            .collect();

        (render_textures, msaa_textures, depth_textures)
    }

    // Helper method to create a uniform bind group for a camera
    fn create_uniform_bind_group(&self, camera: &Camera) -> wgpu::BindGroup {
        let view_proj = camera.get_view_proj_matrix();
        let model = glam::Mat4::IDENTITY;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
        };

        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &self.uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        })
    }
}
