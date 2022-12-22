pub mod additive_renderer;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::memory::allocator::StandardMemoryAllocator;
use bytemuck::{Pod, Zeroable};
use nalgebra::Matrix2x4;
use nalgebra::Vector2;
use crate::job::Job;

pub fn find_rectangle_points(
    center1: Vector2<f32>,
    center2: Vector2<f32>,
    radius: f32,
) -> Matrix2x4<f32> {
    let translated_cent1 = center1 - center2;
    let translated_cent2 = center2 - center1;
    let norm_rad1 = Vector2::normalize(&translated_cent1) * radius;
    let norm_rad2 = Vector2::normalize(&translated_cent2) * radius;

    let point1 = Vector2::new(norm_rad1[1], -norm_rad1[0]) + center2;
    let point2 = Vector2::new(-norm_rad1[1], norm_rad1[0]) + center2;
    let point3 = Vector2::new(norm_rad2[1], -norm_rad2[0]) + center1;
    let point4 = Vector2::new(-norm_rad2[1], norm_rad2[0]) + center1;
    let mut points = vec![point1, point2, point3, point4];
    points.sort_unstable_by(|p1, p2|{
        let mut p1_rad = p1[1].atan2(p1[0]);
        let mut p2_rad = p2[1].atan2(p2[0]);

        if p1_rad < 0.0 { p1_rad += std::f32::consts::PI * 2.0 }
        if p2_rad < 0.0 { p2_rad += std::f32::consts::PI * 2.0 }

        p1_rad.partial_cmp(&p2_rad).unwrap()
    });

    Matrix2x4::from_columns(&points)
}

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct CPUVertex {
    position: [f32; 3],
    color: [u32; 3]
}

vulkano::impl_vertex!(CPUVertex, position);

pub fn process_job(job: Job) {
    let (device , _queue) = additive_renderer::initialize_device();
    let allocator = StandardMemoryAllocator::new_default(device.clone());

    let target_model = job.target_mesh.clone();
    let target_verts = target_model.vertices.iter();

    let mut target_vert_buff: Vec<CPUVertex> = Vec::new();
    for vertex in target_verts {
        let (x, y, z) = (vertex.x, vertex.y, vertex.z);

        let vert = CPUVertex {
            position: [x, y, z],
            color: [0, 0, 255],
        };

        target_vert_buff.push(vert);
    }


    let _ex_buffer = CpuAccessibleBuffer::from_iter(
        &allocator,
        BufferUsage {
            vertex_buffer: true,
            ..Default::default()
        }, 
        false, 
        target_vert_buff);
}