pub mod shaders;
use crate::job::Job;

use std::error::Error;
use std::println;
use std::ptr::null;
use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use nalgebra::Matrix2x4;
use nalgebra::Vector2;
use image::{ImageBuffer, Rgba};
use renderdoc;
use vulkano::LoadingError;
use vulkano::device;
use vulkano::device::DeviceExtensions;
use vulkano::device::physical::PhysicalDevice;
use vulkano::image::SwapchainImage;
use vulkano::instance::InstanceExtensions;
use vulkano::render_pass::RenderPass;
use vulkano::{
    instance::{Instance, InstanceCreateInfo},
    VulkanLibrary,
    device::Queue,
    device::{Device, DeviceCreateInfo, QueueCreateInfo},
    swapchain::{
        Swapchain, SwapchainCreateInfo, acquire_next_image, AcquireError,
        SwapchainCreationError, SwapchainPresentInfo
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    image::view::ImageView,
    image::{StorageImage, ImageUsage, ImageCreateFlags},
    memory::allocator::StandardMemoryAllocator,
    pipeline::graphics::depth_stencil::DepthStencilState,
    pipeline::graphics::input_assembly::InputAssemblyState,
    pipeline::graphics::vertex_input::BuffersDefinition,
    pipeline::graphics::viewport::{Viewport, ViewportState},
    pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint},
    render_pass::{Framebuffer, FramebufferCreateInfo, Subpass},
    sync::{self, GpuFuture},
    buffer::BufferUsage,
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, RenderPassBeginInfo,
        SubpassContents,
    },
    swapchain::Surface
};

use vulkano_win::VkSurfaceBuild;
//use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct CPUVertex {
    in_vert: [f32; 3],
    in_color: [f32; 4],
}

struct GPUInstance {
    spawn_window: bool,
    library: Arc<VulkanLibrary>,
    instance: Arc<Instance>,
    instance_extensions: InstanceExtensions,
    event_loop: Option<EventLoop<()>>,
    surface: Option<Arc<Surface>>,
    device_extensions: DeviceExtensions,
    physical_device: Arc<PhysicalDevice>,
    device: Arc<Device>,
    queue_family_indices: Vec<u32>,
    queues: Vec<Arc<Queue>>,
    viewport: Viewport,
    swapchain: Option<Arc<Swapchain>>,
    swap_images: Option<Vec<Arc<SwapchainImage>>>,
    standard_mem_alloc: StandardMemoryAllocator,
    command_buff_allocator: StandardCommandBufferAllocator,
}

impl GPUInstance {
    pub fn initialize_instance(spawn_window: bool) -> Result<GPUInstance, LoadingError> {
        let library = match VulkanLibrary::new() {
            Ok(library) => library,
            Err(error) => return Err(error),
        };

        let required_instance_extensions = vulkano_win::required_extensions(&library);
        let mut instance_create_info = InstanceCreateInfo {
            application_name: Some(String::from("Thunder Path")),
            enabled_extensions: required_instance_extensions,
            ..Default::default()
        };

        let instance = Instance::new(library, instance_create_info).unwrap();

        let event_loop: Option<EventLoop<()>> = None;
        let surface: Option<Arc<Surface>> = None;
        if spawn_window {
            let event_loop = Some(EventLoop::new());
            let surface = Some(WindowBuilder::new()
                .build_vk_surface(&event_loop.unwrap(), instance)
                .unwrap());
        }
        
        let available_devices = instance
            .enumerate_physical_devices()
            .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let mut graphics_queue: Option<u32> = None;
        let mut presentation_queue: Option<u32> = None;
        let mut compute_queue: Option<u32> = None;
        let mut transfer_queue: Option<u32> = None;
        let mut chosen_physical_device: Option<Arc<PhysicalDevice>> = None;
        for physical_device in available_devices {
            let queues = physical_device.queue_family_properties();
            if physical_device.supported_extensions().contains(&device_extensions) {
                continue;
            }
            for (i, queue) in queues.iter().enumerate() {
                if queue.queue_flags.intersects(device::QueueFlags::GRAPHICS) {
                    graphics_queue = Some(i as u32);
                }
                if queue.queue_flags.intersects(device::QueueFlags::COMPUTE) {
                    compute_queue = Some(i as u32);
                }
                if queue.queue_flags.intersects(device::QueueFlags::TRANSFER) {
                    transfer_queue = Some(i as u32);
                }
                if physical_device.surface_support(i as u32, &surface).unwrap_or(false) {
                    presentation_queue = Some(i as u32);
                }
            }

            if graphics_queue.is_some() && presentation_queue.is_some() &&
                compute_queue.is_some() && transfer_queue.is_some() {
                    chosen_physical_device = Some(physical_device);
                    break;
                } else {
                    graphics_queue = None;
                    presentation_queue = None;
                    compute_queue = None;
                    transfer_queue = None;
                }
        }

        let physical_device = chosen_physical_device
            .expect("no suitable physical device found");

        let queue_indices = vec![graphics_queue.unwrap(), 
                                 presentation_queue.unwrap(),
                                 compute_queue.unwrap(),
                                 transfer_queue.unwrap()];

        let queue_info: Vec<QueueCreateInfo> = Vec::new();
        for i in queue_indices {
            queue_info.push(
                QueueCreateInfo {
                    queue_family_index: i,
                    ..Default::default()
                });
        }

        println!(
            "Using: {} {:?}",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
            );

        let (device, mut queues_iter) = Device::new(physical_device, DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: queue_info,
            ..Default::default()
        }).unwrap();

        let queues = Vec::from_iter(queues_iter.into_iter());

        let mut swapchain: Option<Arc<Swapchain>> = None;
        let mut swap_images: Option<Vec<Arc<SwapchainImage>>> = None;
        if spawn_window {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface.unwrap(), Default::default())
                .unwrap();
            
            let surface_format = Some(
                device
                    .physical_device()
                    .surface_formats(&surface.unwrap(), Default::default())
                    .unwrap()[0]
                    .0,
            );
            let window = surface.unwrap().object()
                .unwrap().downcast_ref::<Window>().unwrap();

            let (some_swapchain, some_swap_images) = Swapchain::new(
                device.clone(), 
                surface.unwrap().clone(), 
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count,
                    image_format: surface_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),

                    ..Default::default()
                }
            ).unwrap();

            swapchain = Some(some_swapchain);
            swap_images = Some(some_swap_images);
        }

        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [0.0, 0.0],
            depth_range: 0.0..1.0,
        };

        let memory_alloc = StandardMemoryAllocator::new_default(device.clone());

        let mut framebuffers = window_size_depen
        

        return Ok(GPUInstance { 
            spawn_window,
            library,
            instance,
            instance_extensions: required_instance_extensions,
            event_loop,
            surface,
            device_extensions,
            physical_device,
            device,
            queue_family_indices: queue_indices,
            queues,
            viewport,
            swapchain,
            swap_images,
            standard_mem_alloc: memory_alloc,
            command_buff_allocator: ()
        });
        
    }


    fn create_gui_renderpass(
        device: Arc<Device>,
        swapchain: Arc<Swapchain>
    ) -> Arc<RenderPass> {
        vulkano::single_pass_renderpass!(
            device,
            attachments: {
                color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },
        },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        );
    }

    /*fn create_gui_framebuffers(
        images: &[Arc<SwapchainImage>],
        rend
    )*/
}

pub fn initialize_device() -> (Arc<vulkano::device::Device>, Arc<Queue>) {
    let v_lib = VulkanLibrary::new().unwrap();

    println!("API Version: {}", v_lib.api_version());

    let create_info = InstanceCreateInfo::default();
    let main_instance = Instance::new(v_lib, create_info).unwrap();

    let dev_desc = main_instance
        .enumerate_physical_devices()
        .unwrap()
        .next()
        .unwrap();

    let queue_family_index = dev_desc
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_, q)| q.queue_flags.graphics)
        .expect("couldn't find a graphical queue family") as u32;

    let queue_families = dev_desc.queue_family_properties();

    let (device, mut queues) = Device::new(
        dev_desc,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("Failed to create device and queues.");

    let queue = queues.next().unwrap();

    return (device, queue);
}

vulkano::impl_vertex!(CPUVertex, in_vert, in_color);

pub fn process_job(job: Job) {
    let mut renderdoc_res = renderdoc::RenderDoc::<renderdoc::V140>::new();

    let (device, queue) = initialize_device();
    let allocator = StandardMemoryAllocator::new_default(device.clone());
    let command_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo {
            ..Default::default()
        },
    );

    renderdoc_res = match renderdoc_res {
        Ok(mut api) => {
            api.start_frame_capture(null(), null());
            Ok(api)
        },

        Err(error) => Err(error),
    };

    let descriptor_allocator = StandardDescriptorSetAllocator::new(device.clone());

    let target_model = job.target_mesh.clone();
    let stock_model = job.target_mesh.clone();
    let (target_vert_buff, target_bounds) = import_verts(target_model);
    let (stock_vert_buff, _stock_bounds) = import_verts(stock_model);
    let target_vert_count: u32 = target_vert_buff.len().try_into().unwrap();
    let _stock_vert_count: u32 = stock_vert_buff.len().try_into().unwrap();

    let ortho_matrix = generate_ortho_matrix(
        target_bounds[0].into(),
        target_bounds[1].into(),
        target_bounds[2].into(),
        target_bounds[3].into(),
        (target_bounds[5] - 10.0).into(),
        target_bounds[4].into(),
    );
    println!("Ortho Matrix: {}", ortho_matrix);

    let ortho_uniform_buffer = CpuAccessibleBuffer::from_data(
        &allocator, 
        BufferUsage {
            uniform_buffer: true,
            ..Default::default()
        }, 
        false, 
        ortho_matrix
    )
    .expect("Failed to create ortho matrix buffer");

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
    let depth_image = StorageImage::with_usage(
        &allocator,
        vulkano::image::ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        vulkano::format::Format::D16_UNORM,
        ImageUsage {
            depth_stencil_attachment: true,
            ..Default::default()
        },
        ImageCreateFlags::empty(),
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
                load: Clear,
                store: DontCare,
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

    let additive_pipeline_builder = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<CPUVertex>())
        .vertex_shader(target_vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(target_frag.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(additive_passes.clone(), 0).unwrap())
        .build(device.clone())
        .expect("Pipeline Building Has Failed");

    //Setup Descriptor Sets
    let layout = additive_pipeline_builder
        .layout()
        .set_layouts()
        .get(0)
        .unwrap();
    let desc_set = PersistentDescriptorSet::new(&descriptor_allocator,
        layout.clone(),
        [WriteDescriptorSet::buffer(0, ortho_uniform_buffer)])
        .unwrap();

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
                clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into()), Some(1.0.into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassContents::Inline,
        )
        .unwrap()
        .bind_pipeline_graphics(additive_pipeline_builder.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            additive_pipeline_builder.layout().clone(),
            0,
            desc_set.clone(),
        )
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

    renderdoc_res = match renderdoc_res {
        Ok(mut api) => {
            api.end_frame_capture(null(), null());
            Ok(api)
        },

        Err(error) => Err(error)
    };

    match renderdoc_res {
        Ok(_) => println!("Capture Data Should have been created."),
        Err(err) => println!("No Data Captured: {}", err),
    }

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
            bounds[0] = x; //Left
        }
        if x > bounds[1] {
            bounds[1] = x; //Right
        }
        if y < bounds[2] {
            bounds[2] = y; //Down
        }
        if y > bounds[3] {
            bounds[3] = y; //Up
        }
        if z < bounds[5] {
            bounds[5] = z; //Near
        }
        if z > bounds[4] {
            bounds[4] = z; //Far
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

// Learned math from 
// https://github.com/PacktPublishing/Vulkan-Cookbook/blob/master/Library/Source%20Files/10%20Helper%20Recipes/05%20Preparing%20an%20orthographic%20projection%20matrix.cpp
fn generate_ortho_matrix(
    left_plane: f32,
    right_plane: f32,
    bottom_plane: f32,
    top_plane: f32,
    near_plane: f32,
    far_plane: f32
) -> nalgebra::Matrix4<f32> {
    let ortho_matrix = nalgebra::Matrix4::new(
        2.0 / (right_plane - left_plane),
        0.0,
        0.0,
        0.0,

        0.0,
        2.0 / (bottom_plane - top_plane),
        0.0,
        0.0,

        0.0,
        0.0,
        1.0 / (near_plane - far_plane),
        0.0,

        -(right_plane + left_plane) / (right_plane - left_plane),
        -(bottom_plane + top_plane) / (bottom_plane - top_plane),
        near_plane / (near_plane - far_plane),
        1.0
    );

    return ortho_matrix.transpose();
}
