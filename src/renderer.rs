use crate::camera::Camera;
use crate::mesh::{Mesh, Vertex};
use crate::texture::Texture;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use std::path::PathBuf;
use wgpu::util::DeviceExt;
use winit::window::Window;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
}

#[allow(dead_code)]
pub struct Renderer {
    // No lifetime parameter
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group: wgpu::BindGroup,
    surface: Option<wgpu::Surface<'static>>, // Use 'static lifetime
    surface_config: Option<wgpu::SurfaceConfiguration>,

    // Image rendering related fields
    width: u32,
    height: u32,
    sample_count: u32,
    background_color: wgpu::Color,
    device_buffers: Vec<wgpu::Buffer>,
    staging_buffers: Vec<wgpu::Buffer>,
    buffer_alignment: u64,
    aligned_bytes_per_row: u32,
    pending_readbacks: Vec<Option<wgpu::BufferSlice<'static>>>,

    // Mesh rendering resources
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    index_count: u32,

    // MSAA and depth texture for window rendering
    msaa_texture: Option<wgpu::Texture>,
    depth_texture: Option<wgpu::Texture>,
}

impl Renderer {
    // New constructor for window-based rendering
    pub async fn new_with_window(
        window: &Window,
        aa_quality: Option<u8>,
        texture_path: Option<&PathBuf>,
        background_color: Option<wgpu::Color>,
    ) -> Result<Self> {
        // Create instance
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = unsafe {
            let surface = instance.create_surface(window)?;
            std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface)
        };

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find a suitable GPU adapter"))?;

        log::info!(
            "Using adapter: {} (backend: {:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        // Request device
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

        // Configure surface
        let window_size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: window_size.width,
            height: window_size.height,
            present_mode: wgpu::PresentMode::Fifo, // VSync
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &surface_config);

        // Determine sample count
        let sample_count = match aa_quality {
            Some(0) => 1,        // No AA
            Some(1) => 2,        // Low
            Some(2) | None => 4, // Medium (default)
            Some(_) => 8,        // High or any other value
        };

        // Check adapter support
        let sample_flags = adapter.get_texture_format_features(format).flags;

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
        let sample_count = sample_count.min(max_sample_count);

        if sample_count > 1 {
            log::info!("Using MSAA with {sample_count} samples");
        } else {
            log::info!("MSAA disabled");
        }

        // Create bind group layouts
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

        // Create default texture
        let texture = match texture_path {
            Some(path) => {
                log::info!("Loading texture from {}", path.display());
                Texture::load(&device, &queue, path)?
            }
            None => {
                log::info!("No texture provided. Using default white texture.");
                Texture::create_default(&device, &queue)
            }
        };

        // Create texture bind group
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

        // Create shaders
        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/vertex.wgsl"
            ))),
        });

        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../shaders/fragment.wgsl"
            ))),
        });

        // Create pipeline
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
                    format,
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
            r: 0.1,
            g: 0.1,
            b: 0.1,
            a: 1.0,
        });

        // Create MSAA and depth textures
        let msaa_texture = if sample_count > 1 {
            Some(create_msaa_texture(
                &device,
                window_size.width,
                window_size.height,
                sample_count,
                format,
            ))
        } else {
            None
        };

        let depth_texture =
            create_depth_texture(&device, window_size.width, window_size.height, sample_count);

        Ok(Self {
            device: device.clone(),
            queue,
            pipeline,
            uniform_bind_group_layout,
            texture_bind_group_layout,
            texture_bind_group,
            surface: Some(surface),
            surface_config: Some(surface_config),
            width: window_size.width,
            height: window_size.height,
            sample_count,
            background_color,
            device_buffers: Vec::new(),
            staging_buffers: Vec::new(),
            pending_readbacks: Vec::new(),
            buffer_alignment: 0,      // Not used for window rendering
            aligned_bytes_per_row: 0, // Not used for window rendering
            vertex_buffer: None,
            index_buffer: None,
            index_count: 0,
            msaa_texture: Some(msaa_texture.unwrap_or_else(|| {
                create_msaa_texture(
                    &device,
                    window_size.width,
                    window_size.height,
                    1, // No MSAA
                    format,
                )
            })),
            depth_texture: Some(depth_texture),
        })
    }

    // Resize the surface and related textures
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return; // Ignore invalid sizes
        }

        if let Some(config) = &mut self.surface_config {
            config.width = width;
            config.height = height;

            if let Some(surface) = &self.surface {
                surface.configure(&self.device, config);

                // Recreate MSAA texture
                if self.sample_count > 1 {
                    self.msaa_texture = Some(create_msaa_texture(
                        &self.device,
                        width,
                        height,
                        self.sample_count,
                        config.format,
                    ));
                }

                // Recreate depth texture
                self.depth_texture = Some(create_depth_texture(
                    &self.device,
                    width,
                    height,
                    self.sample_count,
                ));
            }
        }

        self.width = width;
        self.height = height;
    }

    // Load mesh data to GPU
    pub fn load_mesh(&mut self, mesh: &Mesh) {
        self.vertex_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));

        self.index_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            },
        ));

        self.index_count = u32::try_from(mesh.indices.len()).unwrap();
    }

    // Render to window surface
    pub fn render_to_window(&mut self, mesh: &Mesh, camera: &Camera) -> Result<()> {
        // Ensure mesh data is loaded
        if self.vertex_buffer.is_none() || self.index_buffer.is_none() {
            self.load_mesh(mesh);
        }

        // Get the surface texture
        let surface = self.surface.as_ref().unwrap();
        let frame = surface
            .get_current_texture()
            .map_err(|e| anyhow::anyhow!("Failed to get surface texture: {:?}", e))?;

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Create uniform bind group for this frame
        let uniform_buffer = self.create_uniform_buffer(camera);
        let uniform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &self.uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Get MSAA and depth views
        let msaa_view = self
            .msaa_texture
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = self
            .depth_texture
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Record render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: if self.sample_count > 1 {
                        &msaa_view
                    } else {
                        &view
                    },
                    resolve_target: if self.sample_count > 1 {
                        Some(&view)
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
            render_pass.set_vertex_buffer(0, self.vertex_buffer.as_ref().unwrap().slice(..));
            render_pass.set_index_buffer(
                self.index_buffer.as_ref().unwrap().slice(..),
                wgpu::IndexFormat::Uint32,
            );
            render_pass.draw_indexed(0..self.index_count, 0, 0..1);
        }

        // Submit command buffer
        self.queue.submit(std::iter::once(encoder.finish()));

        // Present the frame
        frame.present();

        Ok(())
    }

    // Helper method to create a uniform buffer for a camera
    fn create_uniform_buffer(&self, camera: &Camera) -> wgpu::Buffer {
        let view_proj = camera.get_view_proj_matrix();
        let model = glam::Mat4::IDENTITY;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
        };

        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }
}

// Helper function to create MSAA texture
fn create_msaa_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MSAA Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

// Helper function to create depth texture
fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}
