use std::sync::Arc;
use std::time::Instant;

use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        BufferUsage,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferUsage, RenderPassBeginInfo, RenderingAttachmentInfo,
        RenderingInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet,
        WriteDescriptorSet,
    },
    device::QueueFlags,
    memory::allocator::MemoryTypeFilter,
    pipeline::{Pipeline, PipelineBindPoint},
    render_pass,
    swapchain::{
        acquire_next_image, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, VulkanError,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use super::shaders;
use super::GPUInstance;
use super::GuiResources;
use super::SceneContents;

pub fn run_gui_loop(
    gpu_instance: Arc<GPUInstance>,
    mut gui_resources: GuiResources,
    mut scene: SceneContents,
    event_loop: EventLoop<()>,
) {
    let queue_index = gpu_instance
        .physical_device
        .queue_family_properties()
        .iter()
        .position(|fam| fam.queue_flags.contains(QueueFlags::GRAPHICS))
        .unwrap();

    let device = gpu_instance.device.clone();
    let graphics_queue = gpu_instance.queues[queue_index as usize].clone();

    //----------------------Constants---------------------------------------//
    let vulkan_adjustment = nalgebra::Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        0.0, 0.0, 0.0, -1.0
    );

    //----------------------Allocators--------------------------------------//
    let standard_alloc = gpu_instance.standard_mem_alloc.clone();

    let command_buffer_allocater =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let descriptor_set_alloc =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());

    // Allocator dedicated for Uniform Buffers
    let ubo_subbuffer_alloc = SubbufferAllocator::new(
        standard_alloc.clone(),
        SubbufferAllocatorCreateInfo {
            buffer_usage: BufferUsage::UNIFORM_BUFFER,
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
    );

    //--------------------Begin Event Loop & Rendering----------------------//
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
                let window = gui_resources
                    .surface
                    .object()
                    .unwrap()
                    .downcast_ref::<Window>()
                    .unwrap();

                let dimensions = window.inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let (new_swapchain, new_images) = gui_resources
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: dimensions.clone().into(),
                            ..gui_resources.swapchain.create_info()
                        })
                        .expect("Failed to recreate swapchain");

                    gui_resources.swapchain = new_swapchain;
                    gui_resources.swapchain_images = new_images.clone();
                    gui_resources.gui_framebuffers =
                        GPUInstance::create_gui_framebuffers(
                            &new_images,
                            &gui_resources.gui_renderpass,
                            &mut gui_resources.viewport,
                            gpu_instance.standard_mem_alloc.clone(),
                        );

                    gui_resources.gui_renderpass =
                        GPUInstance::create_gui_renderpass(
                            device.clone(),
                            gui_resources.swapchain.clone(),
                        );

                    gui_resources.gui_framebuffers =
                        GPUInstance::create_gui_framebuffers(
                            &gui_resources.swapchain_images,
                            &gui_resources.gui_renderpass,
                            &mut gui_resources.viewport,
                            standard_alloc.clone(),
                        );

                    scene.pipelines[0] =
                        gpu_instance.create_gui_mesh_pipeline(&gui_resources);

                    recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(
                        gui_resources.swapchain.clone(),
                        None,
                    )
                    .map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocater,
                    graphics_queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .expect("failed to create command buffer builder");

                let gui_layout = scene.pipelines[0].layout();
                let model_vbo = scene.pipeline_dependencies[0].vbo.clone();
                let desc_layout = gui_layout.set_layouts().get(0).unwrap();
                let view = nalgebra::Matrix4::new_translation(
                    &nalgebra::Vector3::new(0.0, 0.0, 20.0)
                );

                let perspective_mat = nalgebra::Matrix4::new_perspective(
                    dimensions.width as f32 / dimensions.height as f32,
                    30.0,
                    -12.0,
                    40.0,
                );

                let v_ubo_contents = shaders::gui_mesh_vert::MatrixUniforms {
                    model: nalgebra::Matrix4::identity(),
                    view: view,
                    proj: perspective_mat * vulkan_adjustment,
                };

                let mvp_ubo = ubo_subbuffer_alloc.allocate_sized().unwrap();
                *mvp_ubo.write().unwrap() = v_ubo_contents;

                let gui_desc_set = PersistentDescriptorSet::new(
                    &descriptor_set_alloc,
                    desc_layout.clone(),
                    [WriteDescriptorSet::buffer(0, mvp_ubo)],
                    [],
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![
                                Some([0.0, 0.0, 0.0, 1.0].into()),
                                Some(1f32.into()),
                            ],
                            ..RenderPassBeginInfo::framebuffer(
                                gui_resources.gui_framebuffers
                                    [image_index as usize]
                                    .clone(),
                            )
                        },
                        Default::default(),
                    )
                    .unwrap()
                    /*.set_viewport(
                        0,
                        [gui_resources.viewport.clone()].into_iter().collect(),
                    )
                    .unwrap()*/
                    .bind_pipeline_graphics(scene.pipelines[0].clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        gui_layout.clone(),
                        0,
                        gui_desc_set,
                    )
                    .unwrap()
                    .bind_vertex_buffers(0, model_vbo.clone())
                    .unwrap()
                    .draw(model_vbo.len() as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass(Default::default())
                    .unwrap();

                let render_cmd_buf = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(graphics_queue.clone(), render_cmd_buf)
                    .unwrap()
                    .then_swapchain_present(
                        graphics_queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            gui_resources.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end =
                            Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        previous_frame_end =
                            Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    });
}
