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
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
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

            // Process mesh positions, normals, and texture coordinates
            let positions: Vec<[f32; 3]> = mesh
                .positions
                .chunks_exact(3)
                .map(|p| [p[0], p[1], p[2]])
                .collect();

            let normals = if !mesh.normals.is_empty() {
                mesh.normals
                    .chunks_exact(3)
                    .map(|n| [n[0], n[1], n[2]])
                    .collect::<Vec<[f32; 3]>>()
            } else {
                // Generate default normals if none provided
                positions
                    .iter()
                    .map(|_| [0.0, 1.0, 0.0])
                    .collect::<Vec<[f32; 3]>>()
            };

            let tex_coords = if !mesh.texcoords.is_empty() {
                mesh.texcoords
                    .chunks_exact(2)
                    .map(|t| [t[0], t[1]])
                    .collect::<Vec<[f32; 2]>>()
            } else {
                // Generate default UVs if none provided
                positions
                    .iter()
                    .map(|_| [0.0, 0.0])
                    .collect::<Vec<[f32; 2]>>()
            };

            // Create vertices
            for i in 0..positions.len() {
                vertices.push(Vertex {
                    position: positions[i],
                    normal: normals[i],
                    tex_coords: tex_coords[i],
                });
            }

            // Add indices
            for index in mesh.indices {
                indices.push(index as u32 + index_offset);
            }

            index_offset = vertices.len() as u32;
        }

        // Generate vertex normals if not present or we need to recalculate them
        if vertices.iter().all(|v| v.normal == [0.0, 1.0, 0.0]) {
            Self::compute_normals(&mut vertices, &indices);
        }

        Ok(Self { vertices, indices })
    }

    // Compute normals based on face adjacency
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
