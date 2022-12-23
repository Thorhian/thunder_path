pub mod additive_renderer;
pub mod shaders;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use bytemuck::{Pod, Zeroable};
use nalgebra::Matrix2x4;
use nalgebra::Vector2;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::render_pass::Subpass;
use crate::job::Job;


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
    let stock_model = job.target_mesh.clone();
    let (target_vert_buff, target_bounds) = import_verts(target_model);
    let (stock_vert_buff, stock_bounds) = import_verts(stock_model);

    let ortho_matrix = nalgebra::Orthographic3::new(
        target_bounds[0], target_bounds[1],
        target_bounds[2], target_bounds[3],
        target_bounds[4], target_bounds[5]);

    ortho_matrix.as_matrix().iter();

    let gpu_target_buffer = CpuAccessibleBuffer::from_iter(
        &allocator,
        BufferUsage {
            vertex_buffer: true,
            ..Default::default()
        }, 
        false, 
        target_vert_buff
    );

    let gpu_stock_buffer = CpuAccessibleBuffer::from_iter(
        &allocator,
        BufferUsage {
            vertex_buffer: true,
            ..Default::default()
        },
        false,
        stock_vert_buff
    );

    //Load Shaders
    let target_vs = shaders::target_vs::load(device.clone())
        .expect("Failed to Create Target Vertex Shader");

    let additive_pipeline_builder = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<CPUVertex>())
        .vertex_shader(target_vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([fillme]))
        .fragment_shader(, )
        .render_pass(Subpass::from(, ))
        .build(device.clone())
        .expect("Pipeline Building Has Failed");

    

}

fn import_verts(mesh: & russimp::mesh::Mesh)
                -> (Vec<CPUVertex>, Vec<f32>) {
    let vertices = mesh.vertices.iter();
    let first_vert = mesh.vertices
        .first()
        .expect("No Vertices Found in Mesh");

    let mut vertice_buffer: Vec<CPUVertex> = Vec::new();
    let mut bounds: Vec<f32> = vec![first_vert.x, first_vert.x,
                                    first_vert.y, first_vert.y,
                                    first_vert.z, first_vert.z];

    for vertex in vertices {
        let (x, y, z) = (vertex.x, vertex.y, vertex.z);

        if x < bounds[0] {
            bounds[0] = x;
        }
        if x > bounds[1] {
            bounds[1] = x;
        }
        if y < bounds[2] {
            bounds[2] = y;
        }
        if y > bounds[3] {
            bounds[3] = y;
        }
        if z < bounds[5] {
            bounds[5] = z;
        }
        if z > bounds[4] {
            bounds[4] = z;
        }


        let converted_vert = CPUVertex {
            position: [x, y, z],
            color: [0, 0, 255],
        };

        vertice_buffer.push(converted_vert);
    }

    return (vertice_buffer, bounds)

}

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
