use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::{
    path::Path,
    sync::{Arc, Mutex},
    thread,
};

const CHUNK_SIZE: usize = 5000;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl Vertex {
    pub const fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // tex_coords
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn load(path: &Path) -> Result<Self> {
        let (models, _materials) = tobj::load_obj(
            path,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
        )?;

        // For large meshes, process in parallel
        if models.len() > 1 {
            return Self::load_parallel(&models);
        }

        // Combine all models into a single mesh
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut index_offset = 0;

        for model in models {
            let mesh = model.mesh;

            // Process vertices
            let positions_count = mesh.positions.len() / 3;
            vertices.reserve(positions_count);

            for i in 0..positions_count {
                let pos_i = i * 3;
                let tex_i = i * 2;

                // Get position
                let position = [
                    mesh.positions[pos_i],
                    mesh.positions[pos_i + 1],
                    mesh.positions[pos_i + 2],
                ];

                // Get or create normal
                let normal = if pos_i + 2 < mesh.normals.len() {
                    [
                        mesh.normals[pos_i],
                        mesh.normals[pos_i + 1],
                        mesh.normals[pos_i + 2],
                    ]
                } else {
                    [0.0, 1.0, 0.0] // Default normal
                };

                // Get or create texture coordinates
                let tex_coords = if tex_i + 1 < mesh.texcoords.len() {
                    [mesh.texcoords[tex_i], mesh.texcoords[tex_i + 1]]
                } else {
                    [0.0, 0.0] // Default texture coordinates
                };

                vertices.push(Vertex {
                    position,
                    normal,
                    tex_coords,
                });
            }

            // Add indices - pre-allocate to avoid resizing
            let mesh_indices_count = mesh.indices.len();
            indices.reserve(indices.len() + mesh_indices_count);

            for index in &mesh.indices {
                indices.push(*index + index_offset);
            }

            index_offset = u32::try_from(vertices.len()).unwrap();
        }

        // Generate vertex normals if not present
        if vertices.iter().all(|v| v.normal == [0.0, 1.0, 0.0]) {
            Self::compute_normals(&mut vertices, &indices);
        }

        Ok(Self { vertices, indices })
    }

    // Process multiple models in parallel
    fn load_parallel(models: &[tobj::Model]) -> Result<Self> {
        // Create shared data structures
        let vertices = Arc::new(Mutex::new(Vec::new()));
        let indices = Arc::new(Mutex::new(Vec::new()));
        let index_offset = Arc::new(Mutex::new(0u32));

        // Create threads for processing each model
        let mut handles = Vec::new();

        for model in models {
            let model = model.clone();
            let vertices = Arc::clone(&vertices);
            let indices = Arc::clone(&indices);
            let index_offset = Arc::clone(&index_offset);

            let handle = thread::spawn(move || {
                let mesh = model.mesh;
                let positions_count = mesh.positions.len() / 3;

                // Process vertices for this model
                let mut model_vertices = Vec::with_capacity(positions_count);

                for i in 0..positions_count {
                    let pos_i = i * 3;
                    let tex_i = i * 2;

                    let position = [
                        mesh.positions[pos_i],
                        mesh.positions[pos_i + 1],
                        mesh.positions[pos_i + 2],
                    ];

                    let normal = if pos_i + 2 < mesh.normals.len() {
                        [
                            mesh.normals[pos_i],
                            mesh.normals[pos_i + 1],
                            mesh.normals[pos_i + 2],
                        ]
                    } else {
                        [0.0, 1.0, 0.0] // Default normal
                    };

                    let tex_coords = if tex_i + 1 < mesh.texcoords.len() {
                        [mesh.texcoords[tex_i], mesh.texcoords[tex_i + 1]]
                    } else {
                        [0.0, 0.0] // Default texture coordinates
                    };

                    model_vertices.push(Vertex {
                        position,
                        normal,
                        tex_coords,
                    });
                }

                // Get current index offset
                let current_offset = *index_offset.lock().unwrap();

                // Process indices for this model
                let mut model_indices = Vec::with_capacity(mesh.indices.len());
                for index in &mesh.indices {
                    model_indices.push(*index + current_offset);
                }

                // Update global structures
                let new_offset = current_offset + u32::try_from(model_vertices.len()).unwrap();
                *index_offset.lock().unwrap() = new_offset;

                vertices.lock().unwrap().extend(model_vertices);
                indices.lock().unwrap().extend(model_indices);
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread panicked during mesh loading");
        }

        // Get the final mesh data
        let mut vertices = Arc::try_unwrap(vertices).unwrap().into_inner().unwrap();
        let indices = Arc::try_unwrap(indices).unwrap().into_inner().unwrap();

        // Generate vertex normals if not present
        if vertices.iter().all(|v| v.normal == [0.0, 1.0, 0.0]) {
            Self::compute_normals(&mut vertices, &indices);
        }

        Ok(Self { vertices, indices })
    }

    pub fn calculate_bounding_box(&self) -> (glam::Vec3, glam::Vec3) {
        // Initialize with extreme values
        let mut min = glam::Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = glam::Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        // For large meshes, use parallel computation
        if self.vertices.len() > 10000 {
            return self.calculate_bounding_box_parallel();
        }

        // Find min and max points
        for vertex in &self.vertices {
            let pos = glam::Vec3::new(vertex.position[0], vertex.position[1], vertex.position[2]);

            min = min.min(pos);
            max = max.max(pos);
        }

        (min, max)
    }

    // Parallel version of bounding box calculation for large meshes
    fn calculate_bounding_box_parallel(&self) -> (glam::Vec3, glam::Vec3) {
        const CHUNK_SIZE: usize = 5000;
        let chunk_count = (self.vertices.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;

        let min_max = Arc::new(Mutex::new(vec![
            (
                glam::Vec3::new(f32::MAX, f32::MAX, f32::MAX),
                glam::Vec3::new(f32::MIN, f32::MIN, f32::MIN)
            );
            chunk_count
        ]));

        let mut handles = vec![];

        // Process chunks in parallel
        for (chunk_index, chunk) in self.vertices.chunks(CHUNK_SIZE).enumerate() {
            let min_max = Arc::clone(&min_max);
            let chunk = chunk.to_vec();

            let handle = thread::spawn(move || {
                let mut chunk_min = glam::Vec3::new(f32::MAX, f32::MAX, f32::MAX);
                let mut chunk_max = glam::Vec3::new(f32::MIN, f32::MIN, f32::MIN);

                for vertex in &chunk {
                    let pos =
                        glam::Vec3::new(vertex.position[0], vertex.position[1], vertex.position[2]);
                    chunk_min = chunk_min.min(pos);
                    chunk_max = chunk_max.max(pos);
                }

                let mut results = min_max.lock().unwrap();
                results[chunk_index] = (chunk_min, chunk_max);
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle
                .join()
                .expect("Thread panicked during bounding box calculation");
        }

        // Combine results from all chunks
        let results = Arc::try_unwrap(min_max).unwrap().into_inner().unwrap();
        let mut global_min = glam::Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut global_max = glam::Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        for (min, max) in results {
            global_min = global_min.min(min);
            global_max = global_max.max(max);
        }

        (global_min, global_max)
    }

    fn compute_normals(vertices: &mut [Vertex], indices: &[u32]) {
        const CHUNK_SIZE: usize = 5000;

        // For small meshes, use the original implementation
        if indices.len() < CHUNK_SIZE * 3 {
            Self::compute_normals_sequential(vertices, indices);
            return;
        }

        // For large meshes, use parallel implementation
        Self::compute_normals_parallel(vertices, indices);
    }

    // Original sequential implementation
    fn compute_normals_sequential(vertices: &mut [Vertex], indices: &[u32]) {
        // Reset all normals
        for vertex in vertices.iter_mut() {
            vertex.normal = [0.0, 0.0, 0.0];
        }

        // Calculate normals for each face
        for chunk in indices.chunks_exact(3) {
            let [i0, i1, i2] = [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize];

            let v0 = Vec3::from(vertices[i0].position);
            let v1 = Vec3::from(vertices[i1].position);
            let v2 = Vec3::from(vertices[i2].position);

            let e1 = v1 - v0;
            let e2 = v2 - v0;
            let normal = e1.cross(e2).normalize();

            // Add normal to all vertices of this face
            for &i in &[i0, i1, i2] {
                vertices[i].normal[0] += normal.x;
                vertices[i].normal[1] += normal.y;
                vertices[i].normal[2] += normal.z;
            }
        }

        // Normalize all normals
        for vertex in vertices.iter_mut() {
            let n = Vec3::from(vertex.normal).normalize();
            vertex.normal = [n.x, n.y, n.z];
        }
    }

    // Parallel implementation for larger meshes
    fn compute_normals_parallel(vertices: &mut [Vertex], indices: &[u32]) {
        // Reset all normals
        for vertex in vertices.iter_mut() {
            vertex.normal = [0.0, 0.0, 0.0];
        }

        let vertex_normals = Arc::new(Mutex::new(vec![Vec3::ZERO; vertices.len()]));
        let mut handles = vec![];

        // Process face normals in parallel chunks
        for chunk in indices.chunks(CHUNK_SIZE * 3) {
            let vertex_normals = Arc::clone(&vertex_normals);
            let vertices = vertices.to_vec(); // Clone for thread safety
            let chunk = chunk.to_vec();

            let handle = thread::spawn(move || {
                let mut local_normals = vec![Vec3::ZERO; vertices.len()];

                for face_indices in chunk.chunks_exact(3) {
                    let [i0, i1, i2] = [
                        face_indices[0] as usize,
                        face_indices[1] as usize,
                        face_indices[2] as usize,
                    ];

                    let v0 = Vec3::from(vertices[i0].position);
                    let v1 = Vec3::from(vertices[i1].position);
                    let v2 = Vec3::from(vertices[i2].position);

                    let e1 = v1 - v0;
                    let e2 = v2 - v0;
                    let normal = e1.cross(e2).normalize();

                    // Add to local normals array
                    for &i in &[i0, i1, i2] {
                        local_normals[i] += normal;
                    }
                }

                // Add local results to global results
                let mut global_normals = vertex_normals.lock().unwrap();
                for (i, local_normal) in local_normals.iter().enumerate() {
                    if *local_normal != Vec3::ZERO {
                        global_normals[i] += *local_normal;
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle
                .join()
                .expect("Thread panicked during normal computation");
        }

        // Normalize and update the vertex normals
        let global_normals = Arc::try_unwrap(vertex_normals)
            .unwrap()
            .into_inner()
            .unwrap();

        for (i, vertex) in vertices.iter_mut().enumerate() {
            let n = global_normals[i].normalize();
            vertex.normal = [n.x, n.y, n.z];
        }
    }
}
