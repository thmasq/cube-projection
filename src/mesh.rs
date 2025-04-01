use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::path::Path;

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

        // Combine all models into a single mesh
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut index_offset = 0;

        for model in models {
            let mesh = model.mesh;

            // Process vertices
            let positions_count = mesh.positions.len() / 3;

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

            // Add indices
            for index in &mesh.indices {
                indices.push({ *index } + index_offset);
            }

            index_offset = u32::try_from(vertices.len()).unwrap();
        }

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

        // Find min and max points
        for vertex in &self.vertices {
            let pos = glam::Vec3::new(vertex.position[0], vertex.position[1], vertex.position[2]);

            min = min.min(pos);
            max = max.max(pos);
        }

        (min, max)
    }

    fn compute_normals(vertices: &mut [Vertex], indices: &[u32]) {
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
}
