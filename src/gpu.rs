pub mod shaders;
pub mod window;

use nalgebra::Matrix2x4;
use nalgebra::Vector2;
use nalgebra::Vector3;
use std::collections::HashMap;
use std::println;
use std::sync::Arc;
use vulkano::image::Image;
use vulkano::pipeline::graphics::color_blend::ColorBlendAttachmentState;
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::VertexDefinition;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::DynamicState;
use vulkano::pipeline::PipelineLayout;
use vulkano::pipeline::PipelineShaderStageCreateInfo;
use winit::event_loop::EventLoop;
use winit::window::Window;
use winit::window::WindowBuilder;
//use renderdoc;

use vulkano::buffer::Subbuffer;
use vulkano::{
    buffer::BufferContents,
    command_buffer::allocator::StandardCommandBufferAllocator,
    /*command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo,
        SubpassContents,
    },*/
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    //device,
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions,
        Queue, QueueCreateInfo, QueueFlags,
    },
    //descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    image::view::ImageView,
    image::ImageUsage,
    instance::InstanceExtensions,
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::FreeListAllocator,
    memory::allocator::GenericMemoryAllocator,
    memory::allocator::StandardMemoryAllocator,
    //pipeline::graphics::depth_stencil::DepthStencilState,
    //pipeline::graphics::input_assembly::InputAssemblyState,
    pipeline::graphics::vertex_input::Vertex,
    pipeline::graphics::viewport::{Viewport, ViewportState},
    pipeline::{GraphicsPipeline, Pipeline},
    //render_pass,
    render_pass::RenderPass,
    render_pass::{Framebuffer, FramebufferCreateInfo, Subpass},
    swapchain::Surface,
    swapchain::{
        //acquire_next_image,
        //AcquireError,
        Swapchain,
        SwapchainCreateInfo,
        //SwapchainCreationError, //SwapchainPresentInfo
    },
    //sync::{self, GpuFuture},
    LoadingError,
    VulkanLibrary,
};

use vulkano_win::VkSurfaceBuild;

#[repr(C)]
#[derive(BufferContents, Vertex)]
pub struct ModelVertex {
    #[format(R32G32B32_SFLOAT)]
    in_vert: [f32; 3],

    #[format(R32G32B32A32_SFLOAT)]
    in_color: [f32; 4],
}

pub struct GuiResources {
    surface: Arc<Surface>,
    viewport: Viewport,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<Image>>,
    gui_renderpass: Arc<RenderPass>,
    gui_framebuffers: Vec<Arc<Framebuffer>>,
}

pub struct GPUInstance {
    library: Arc<VulkanLibrary>,
    pub instance: Arc<Instance>,
    instance_extensions: InstanceExtensions,
    device_extensions: DeviceExtensions,
    physical_device: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub queue_family_indices: HashMap<u32, QueueFlags>,
    pub queues: Vec<Arc<Queue>>,
    pub standard_mem_alloc: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    pub command_buff_allocator: StandardCommandBufferAllocator,
    pub descriptor_allocator: StandardDescriptorSetAllocator,
}

impl GPUInstance {
    pub fn initialize_instance(
        spawn_window: bool,
    ) -> Result<
        (GPUInstance, Option<EventLoop<()>>, Option<GuiResources>),
        LoadingError,
    > {
        //Get The Vulkan Library
        let library = match VulkanLibrary::new() {
            Ok(library) => library,
            Err(error) => return Err(error),
        };

        //If we are using the Gui, get the event loop
        let event_loop: Option<EventLoop<()>> = if spawn_window {
            Some(EventLoop::new())
        } else {
            None
        };

        let mut required_instance_extensions = InstanceExtensions {
            ..Default::default()
        };
        println!("Default Extensions: {:#?}", required_instance_extensions);

        let instance_create_info = InstanceCreateInfo {
            application_name: Some(String::from("Thunder Path")),
            enabled_extensions: required_instance_extensions,
            ..Default::default()
        };

        let instance =
            Instance::new(library.clone(), instance_create_info).unwrap();

        let available_devices = instance.enumerate_physical_devices().unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let surface = if spawn_window {
            Some(
                WindowBuilder::new()
                    .build_vk_surface(&event_loop.unwrap(), instance.clone())
                    .unwrap(),
            )
        } else {
            None
        };

        let mut graphics_queue: Option<u32> = None;
        let mut presentation_queue: Option<u32> = None;
        let mut compute_queue: Option<u32> = None;
        let mut transfer_queue: Option<u32> = None;
        let mut chosen_physical_device: Option<Arc<PhysicalDevice>> = None;
        for physical_device in available_devices {
            let queue_fams = physical_device.queue_family_properties();
            if !physical_device
                .supported_extensions()
                .contains(&device_extensions)
            {
                continue;
            }
            println!("Queue Family: {:#?}", queue_fams);
            for (i, queue_fam) in queue_fams.iter().enumerate() {
                if queue_fam.queue_flags.intersects(QueueFlags::GRAPHICS) {
                    graphics_queue = Some(i as u32);
                }
                if queue_fam.queue_flags.intersects(QueueFlags::COMPUTE) {
                    compute_queue = Some(i as u32);
                }
                if queue_fam.queue_flags.intersects(QueueFlags::TRANSFER) {
                    transfer_queue = Some(i as u32);
                }
                if spawn_window
                    && physical_device
                        .surface_support(
                            i as u32,
                            &surface.as_ref().unwrap().clone(),
                        )
                        .unwrap_or(false)
                {
                    presentation_queue = Some(i as u32);
                }
            }

            if graphics_queue.is_some()
                && (presentation_queue.is_some() || !spawn_window)
                && compute_queue.is_some()
                && transfer_queue.is_some()
            {
                chosen_physical_device = Some(physical_device);
                break;
            } else {
                graphics_queue = None;
                presentation_queue = None;
                compute_queue = None;
                transfer_queue = None;
            }
        }

        let physical_device =
            chosen_physical_device.expect("no suitable physical device found");

        let mut queue_indices = vec![
            (graphics_queue.unwrap(), QueueFlags::GRAPHICS),
            (compute_queue.unwrap(), QueueFlags::COMPUTE),
            (transfer_queue.unwrap(), QueueFlags::TRANSFER),
        ];

        if spawn_window {
            queue_indices
                .push((presentation_queue.unwrap(), QueueFlags::GRAPHICS));
        }

        let mut queue_fam_indices: HashMap<u32, QueueFlags> = HashMap::new();
        for pair in queue_indices {
            match queue_fam_indices.get(&pair.0) {
                Some(old_type) => {
                    queue_fam_indices.insert(pair.0, pair.1.union(*old_type))
                }
                None => queue_fam_indices.insert(pair.0, pair.1),
            };
        }

        let mut queue_info: Vec<QueueCreateInfo> = Vec::new();
        for (i, _) in queue_fam_indices {
            queue_info.push(QueueCreateInfo {
                queue_family_index: i,
                ..Default::default()
            });
        }

        println!("Queue Creation Info Structs: {:#?}", queue_info);

        println!(
            "Using: {} {:?}",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, queues_iter) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: queue_info,
                ..Default::default()
            },
        )
        .unwrap();

        let queues = Vec::from_iter(queues_iter.into_iter());

        let memory_alloc =
            Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let command_buff_allocator = StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        );

        let descriptor_allocator = StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        );

        let event_loop = if spawn_window {
            Some(EventLoop::new())
        } else {
            None
        };

        //Generate GUI data if desired, return optional struct
        let gui_resources = if spawn_window {
            let surface = surface.unwrap();

            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface.clone(), Default::default())
                .unwrap();

            let surface_format = device
                .physical_device()
                .surface_formats(&surface.clone(), Default::default())
                .unwrap()[0]
                .0;

            let window =
                surface.object().unwrap().downcast_ref::<Window>().unwrap();

            let (swapchain, swapchain_images) = Swapchain::new(
                device.clone(),
                surface.clone(),
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
                },
            )
            .unwrap();

            let mut viewport = Viewport {
                offset: [0.0, 0.0],
                extent: [0.0, 0.0],
                depth_range: 0.0..=1.0,
            };

            let gui_renderpass =
                Self::create_gui_renderpass(device.clone(), swapchain.clone());

            let gui_framebuffers = Self::create_gui_framebuffers(
                &swapchain_images,
                &gui_renderpass,
                &mut viewport,
            );

            Some(GuiResources {
                surface,
                viewport,
                swapchain,
                swapchain_images,
                gui_renderpass,
                gui_framebuffers,
            })
        } else {
            None
        };

        return Ok((
            GPUInstance {
                library,
                instance,
                instance_extensions: required_instance_extensions,
                device_extensions,
                physical_device,
                device,
                queue_family_indices: queue_fam_indices,
                queues,
                standard_mem_alloc: memory_alloc,
                command_buff_allocator,
                descriptor_allocator,
            },
            event_loop,
            gui_resources,
        ));
    }

    // Creates renderpass for demo UI
    fn create_gui_renderpass(
        device: Arc<Device>,
        swapchain: Arc<Swapchain>,
    ) -> Arc<RenderPass> {
        return vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
            },
        },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();
    }

    pub fn create_gui_framebuffers(
        images: &[Arc<Image>],
        render_pass: &Arc<RenderPass>,
        viewport: &mut Viewport,
    ) -> Vec<Arc<Framebuffer>> {
        let extent = images.first().unwrap().extent();

        viewport.extent = [extent[0] as f32, extent[1] as f32];

        images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<Arc<Framebuffer>>>()
    }

    pub fn gather_layouts(
        scene: &SceneContents,
        desc_alloc: &StandardDescriptorSetAllocator,
    ) {
        let layout = scene.pipelines[0].layout().set_layouts().get(0).unwrap();

        /*let desc_set = PersistentDescriptorSet::new(
            desc_alloc,
            layout.clone(),
            [WriteDescriptorSet::buffer(0, buffer)]
        );*/
    }

    pub fn create_gui_mesh_pipeline(
        &self,
        gui_resources: &GuiResources,
    ) -> Arc<GraphicsPipeline> {
        // TODO: Will need to redo layout stuff here
        let pipeline = {
            let v_shader = shaders::gui_mesh_vert::load(self.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let f_shader = shaders::gui_mesh_frag::load(self.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let vertex_input_state = ModelVertex::per_vertex()
                .definition(&v_shader.info().input_interface)
                .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(v_shader),
                PipelineShaderStageCreateInfo::new(f_shader),
            ];

            let test_layout = PipelineLayout::new(
                self.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.device.clone())
                    .unwrap(),
            )
            .unwrap();

            let gui_renderpass = gui_resources.gui_renderpass.clone();
            let subpass = Subpass::from(gui_renderpass, 0).unwrap();

            GraphicsPipeline::new(
                self.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(
                        ColorBlendState::with_attachment_states(
                            subpass.num_color_attachments(),
                            ColorBlendAttachmentState::default(),
                        ),
                    ),
                    dynamic_state: [DynamicState::Viewport]
                        .into_iter()
                        .collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(test_layout)
                },
            )
            .unwrap()
        };
        //End Creation of pipeline
        return pipeline;
    }
}
pub enum MeshType {
    TargetMesh,
    StockMesh,
    ObstacleMesh,
}

pub struct MeshModel {
    pub vbo_contents: Vec<ModelVertex>,
    pub mesh_type: MeshType,
    pub bounds: [f32; 6],
}

pub struct PipelineDependencies {
    pub vbo: Subbuffer<[ModelVertex]>,
}

pub struct SceneContents {
    pub pipeline_dependencies: Vec<PipelineDependencies>,
    pub pipelines: Vec<Arc<GraphicsPipeline>>,
    pub mesh_pipe_indices: Vec<usize>,
}

/*pub fn initialize_device() -> (Arc<vulkano::device::Device>, Arc<Queue>) {
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
}*/

/*pub fn process_job(job: Job) {
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


}*/

pub fn import_verts(
    mesh: &russimp::mesh::Mesh,
    color: Vector3<f32>,
) -> (Vec<ModelVertex>, [f32; 6]) {
    let vertices = mesh.vertices.iter();
    let first_vert = mesh.vertices.first().expect("No Vertices Found in Mesh");

    let mut vertice_buffer: Vec<ModelVertex> = Vec::new();
    let mut bounds: [f32; 6] = [
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

        let converted_vert = ModelVertex {
            in_vert: [x, y, z],
            in_color: [color.x, color.y, color.z, 1.0],
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
    far_plane: f32,
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
        1.0,
    );

    return ortho_matrix.transpose();
}
