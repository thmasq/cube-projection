use crate::camera::Camera;
use crate::mesh::{Mesh, Vertex};
use crate::texture::Texture;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::{Arc, Mutex};
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
        let texture = Texture::create_default(&device, &queue);

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

    pub fn init_readback_buffers(&mut self, face_count: usize) {
        let output_buffer_size = u64::from(self.height) * u64::from(self.aligned_bytes_per_row);

        // Create optimal device buffers (for fast GPU writes)
        self.device_buffers = (0..face_count)
            .map(|i| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Device Buffer {i}")),
                    size: output_buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect();

        // Create staging buffers (for CPU reads)
        self.staging_buffers = (0..face_count)
            .map(|i| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Staging Buffer {i}")),
                    size: output_buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            })
            .collect();

        // Initialize pending readbacks vector
        self.pending_readbacks = vec![None; face_count];
    }

    pub async fn render_cube_faces(
        &mut self,
        mesh: &Mesh,
        cameras: &[Camera; 6],
    ) -> Result<Vec<Vec<u8>>> {
        let resource_start = std::time::Instant::now();

        // Ensure we have the buffers
        if self.device_buffers.is_empty() {
            self.init_readback_buffers(cameras.len());
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

        let texture_extent = wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        };

        // Create textures for all faces
        let (render_textures, msaa_textures, depth_textures) =
            self.create_face_textures(cameras.len());

        let resource_prep_time = resource_start.elapsed();
        let render_start = std::time::Instant::now();

        // Process each face with overlapped rendering and readback
        for (i, camera) in cameras.iter().enumerate() {
            // Create a command encoder for this face
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("Face {i} Encoder")),
                });

            // Create uniform buffer
            let uniform_buffer = self.create_uniform_buffer(camera);

            // Create uniform bind group for this camera
            let uniform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Uniform Bind Group"),
                layout: &self.uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            // Get the textures for this face
            let texture_view =
                render_textures[i].create_view(&wgpu::TextureViewDescriptor::default());
            let msaa_view = msaa_textures
                .get(i)
                .map(|tex| tex.create_view(&wgpu::TextureViewDescriptor::default()));
            let depth_view = depth_textures[i].create_view(&wgpu::TextureViewDescriptor::default());

            // Record render pass
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(&format!("Render Pass {i}")),
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

            // Copy from texture to device buffer (optimal for GPU)
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &render_textures[i],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.device_buffers[i],
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(self.aligned_bytes_per_row),
                        rows_per_image: Some(self.height),
                    },
                },
                texture_extent,
            );

            // Copy from device buffer to staging buffer (optimal for CPU)
            encoder.copy_buffer_to_buffer(
                &self.device_buffers[i],
                0,
                &self.staging_buffers[i],
                0,
                u64::from(self.height) * u64::from(self.aligned_bytes_per_row),
            );

            // Submit commands for this face immediately
            self.queue.submit(std::iter::once(encoder.finish()));

            // Start mapping this buffer asynchronously while next face renders
            let buffer_slice = self.staging_buffers[i].slice(..);

            // Use unsafe to extend the lifetime of the buffer slice
            // This is safe because we ensure the buffer outlives this borrow
            let buffer_slice = unsafe {
                std::mem::transmute::<wgpu::BufferSlice<'_>, wgpu::BufferSlice<'static>>(
                    buffer_slice,
                )
            };

            buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

            // Store the pending readback
            self.pending_readbacks[i] = Some(buffer_slice);
        }

        let render_time = render_start.elapsed();
        let readback_start = std::time::Instant::now();

        // Poll until all buffers are mapped
        self.device.poll(wgpu::Maintain::Wait);

        // Process the mapped data in parallel using rayon
        let width = self.width;
        let height = self.height;
        let aligned_bytes_per_row = self.aligned_bytes_per_row;

        let results = Arc::new(Mutex::new(vec![Vec::new(); cameras.len()]));

        rayon::scope(|s| {
            for (i, buffer_slice_opt) in self.pending_readbacks.iter_mut().enumerate() {
                if let Some(buffer_slice) = buffer_slice_opt.take() {
                    let results_clone = Arc::clone(&results);

                    s.spawn(move |_| {
                        // Get the buffer data
                        let data = buffer_slice.get_mapped_range();
                        let output_size = (width * height * 4) as usize;
                        let mut face_data = Vec::with_capacity(output_size);

                        // OPTIMIZATION #4: Efficient padding removal with unsafe
                        unsafe {
                            let src_ptr: *const u8 = data.as_ptr();
                            face_data.set_len(output_size);
                            let dest_ptr: *mut u8 = face_data.as_mut_ptr();

                            for row in 0..height {
                                std::ptr::copy_nonoverlapping(
                                    src_ptr.add((row * aligned_bytes_per_row) as usize),
                                    dest_ptr.add((row * width * 4) as usize),
                                    (width * 4) as usize,
                                );
                            }
                        }

                        // Release the mapped range
                        drop(data);

                        // Store the result
                        results_clone.lock().unwrap()[i] = face_data;
                    });
                }
            }
        });

        // Unmap all buffers
        for buffer in &self.staging_buffers {
            buffer.unmap();
        }

        let readback_time = readback_start.elapsed();

        // Log detailed metrics
        log::info!("Rendering metrics:");
        log::info!("  Resource preparation: {resource_prep_time:.2?}");
        log::info!("  Render commands encoding: {render_time:.2?}");
        log::info!("  GPU execution and buffer readback: {readback_time:.2?}");
        log::info!(
            "  Total rendering: {:.2?}",
            resource_prep_time + render_time + readback_time
        );

        // Return the collected results
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
                    label: Some(&format!("Render Texture {i}")),
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
                        label: Some(&format!("Multisampled Texture {i}")),
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
                    label: Some(&format!("Depth Texture {i}")),
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
