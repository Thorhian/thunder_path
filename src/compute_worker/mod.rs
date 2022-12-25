pub mod additive_renderer;
pub mod shaders;

use std::collections::BTreeMap;

use bytemuck::{Pod, Zeroable};
use nalgebra::Matrix2x4;
use nalgebra::Vector2;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, RenderPassBeginInfo,
    SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::descriptor_set::{self, DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::image::view::ImageView;
use vulkano::image::StorageImage;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, Subpass};
use vulkano::sync::{self, GpuFuture};

use image::{ImageBuffer, Rgba};

use crate::job::Job;

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct CPUVertex {
    in_vert: [f32; 3],
    in_color: [f32; 4],
}

vulkano::impl_vertex!(CPUVertex, in_vert, in_color);

pub fn process_job(job: Job) {
    let (device, queue) = additive_renderer::initialize_device();
    let allocator = StandardMemoryAllocator::new_default(device.clone());
    let command_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo {
            ..Default::default()
        },
    );

    let descriptor_allocator = StandardDescriptorSetAllocator::new(device.clone());

    let target_model = job.target_mesh.clone();
    let stock_model = job.target_mesh.clone();
    let (target_vert_buff, target_bounds) = import_verts(target_model);
    let (stock_vert_buff, _stock_bounds) = import_verts(stock_model);
    let target_vert_count: u32 = target_vert_buff.len().try_into().unwrap();
    let _stock_vert_count: u32 = stock_vert_buff.len().try_into().unwrap();

    let ortho_matrix = nalgebra::Orthographic3::new(
        target_bounds[0],
        target_bounds[1],
        target_bounds[2],
        target_bounds[3],
        target_bounds[4],
        target_bounds[5],
    );

    ortho_matrix.as_matrix().iter();

    let gpu_target_buffer = CpuAccessibleBuffer::from_iter(
        &allocator,
        BufferUsage {
            vertex_buffer: true,
            ..Default::default()
        },
        false,
        target_vert_buff,
    )
    .expect("Failed to create gpu_target_buffer");

    let _gpu_stock_buffer = CpuAccessibleBuffer::from_iter(
        &allocator,
        BufferUsage {
            vertex_buffer: true,
            ..Default::default()
        },
        false,
        stock_vert_buff,
    )
    .expect("Failed to create gpu_stock_buffer");

    //Allocate Image Target
    let image = StorageImage::new(
        &allocator,
        vulkano::image::ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        vulkano::format::Format::R8G8B8A8_UNORM,
        Some(queue.queue_family_index()),
    )
    .unwrap();

    let depth_image = StorageImage::new(
        &allocator,
        vulkano::image::ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        vulkano::format::Format::D16_UNORM,
        Some(queue.queue_family_index()),
    )
    .unwrap();

    let additive_passes = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: vulkano::format::Format::R8G8B8A8_UNORM,
                samples: 1,
        },
            depth: {
                load:Clear,
                store: Store,
                format: vulkano::format::Format::D16_UNORM,
                samples: 1,
            }
    },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
    .unwrap();

    let view = ImageView::new_default(image.clone()).unwrap();
    let depth_view = ImageView::new_default(depth_image.clone()).unwrap();
    let framebuffer = Framebuffer::new(
        additive_passes.clone(),
        FramebufferCreateInfo {
            attachments: vec![view, depth_view],
            ..Default::default()
        },
    )
    .unwrap();

    let mut command_buff_builder = AutoCommandBufferBuilder::primary(
        &command_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [1024.0, 1024.0],
        depth_range: 0.0..1.0,
    };

    //Load Shaders
    let target_vs =
        shaders::target_vs::load(device.clone()).expect("Failed to Create Target Vertex Shader");
    let target_frag = shaders::target_frag::load(device.clone())
        .expect("Failed to Create Target Fragment Shader");
    let _edge_detection = shaders::edge_detection::load(device.clone())
        .expect("Failed to Create Edge Detection Shader");
    let _edge_expansion = shaders::edge_expansion::load(device.clone())
        .expect("Failed to Create Edge Expansion Shader");

    let target_frag_entry = target_frag.entry_point("main").unwrap();

    let additive_pipeline_builder = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<CPUVertex>())
        .vertex_shader(target_vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(target_frag.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(additive_passes.clone(), 0).unwrap())
        .build(device.clone())
        .expect("Pipeline Building Has Failed");

    //Setup Descriptor Sets
    let layout = additive_pipeline_builder
        .layout()
        .set_layouts()
        .get(0)
        .unwrap();
    let desc_set = PersistentDescriptorSet::new(&descriptor_allocator, layout.clone(), []).unwrap();

    let png_buffer = CpuAccessibleBuffer::from_iter(
        &allocator,
        BufferUsage {
            transfer_dst: true,
            ..Default::default()
        },
        false,
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .expect("Failed to create PNG buffer");

    command_buff_builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.2, 0.2, 0.1, 1.0].into()), Some(0.0.into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassContents::Inline,
        )
        .unwrap()
        .bind_pipeline_graphics(additive_pipeline_builder.clone())
        /*.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            additive_pipeline_builder.layout().clone(),
            0,
            desc_set.clone(),
        )*/
        .bind_vertex_buffers(0, gpu_target_buffer.clone())
        //.bind_descriptor_sets()
        .draw(target_vert_count, 1, 0, 0)
        .unwrap()
        .end_render_pass()
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image,
            png_buffer.clone(),
        ))
        .unwrap();

    let command_buff = command_buff_builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buff)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let png_content = png_buffer.read().unwrap();
    let png = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &png_content[..]).unwrap();
    png.save("test_img.png").unwrap();

    println!("Rendering Finished")
}

fn import_verts(mesh: &russimp::mesh::Mesh) -> (Vec<CPUVertex>, Vec<f32>) {
    let vertices = mesh.vertices.iter();
    let first_vert = mesh.vertices.first().expect("No Vertices Found in Mesh");

    let mut vertice_buffer: Vec<CPUVertex> = Vec::new();
    let mut bounds: Vec<f32> = vec![
        first_vert.x,
        first_vert.x,
        first_vert.y,
        first_vert.y,
        first_vert.z,
        first_vert.z,
    ];

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
            in_vert: [x, y, z],
            in_color: [0.0, 0.0, 1.0, 1.0],
        };

        vertice_buffer.push(converted_vert);
    }

    return (vertice_buffer, bounds);
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
    points.sort_unstable_by(|p1, p2| {
        let mut p1_rad = p1[1].atan2(p1[0]);
        let mut p2_rad = p2[1].atan2(p2[0]);

        if p1_rad < 0.0 {
            p1_rad += std::f32::consts::PI * 2.0
        }
        if p2_rad < 0.0 {
            p2_rad += std::f32::consts::PI * 2.0
        }

        p1_rad.partial_cmp(&p2_rad).unwrap()
    });

    Matrix2x4::from_columns(&points)
}
