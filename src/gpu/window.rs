use std::sync::Arc;

use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferUsage, RenderPassBeginInfo, SubpassContents,
    },
    device::QueueFlags,
    swapchain::{
        acquire_next_image, AcquireError, SwapchainCreateInfo,
        SwapchainCreationError,
    },
    sync::{self, GpuFuture},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use super::GPUInstance;
use super::GuiResources;
use super::SceneContents;

pub fn run_gui_loop(
    gpu_instance: Arc<GPUInstance>,
    event_loop: EventLoop<()>,
    mut gui_resources: GuiResources,
    scene: SceneContents,
) {
    let gfx_queue_indice = gpu_instance.queue_family_indices[0];
    assert!(gfx_queue_indice.1 == QueueFlags::GRAPHICS);
    let graphics_queue =
        gpu_instance.queues[gfx_queue_indice.0 as usize].clone();

    let command_buffer_allocater = StandardCommandBufferAllocator::new(
        gpu_instance.device.clone(),
        Default::default(),
    );

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(gpu_instance.device.clone()));
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
                    let (new_swapchain, new_images) = match gui_resources
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: dimensions.into(),
                            ..gui_resources.swapchain.create_info()
                        }) {
                        Ok(r) => r,
                        Err(
                            SwapchainCreationError::ImageExtentNotSupported {
                                ..
                            },
                        ) => return,
                        Err(e) => panic!("failed to recreate swapchain: {e}"),
                    };

                    gui_resources.swapchain = new_swapchain;
                    gui_resources.swapchain_images = new_images.clone();
                    gui_resources.gui_framebuffers =
                        GPUInstance::create_gui_framebuffers(
                            &new_images,
                            &gui_resources.gui_renderpass,
                            &mut gui_resources.viewport,
                        );

                    recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(
                        gui_resources.swapchain.clone(),
                        None,
                    ) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
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

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some(
                                [0.1, 0.1, 0.1, 1.0].into(),
                            )],
                            ..RenderPassBeginInfo::framebuffer(
                                gui_resources.gui_framebuffers
                                    [image_index as usize]
                                    .clone(),
                            )
                        },
                        SubpassContents::Inline,
                    )
                    .unwrap()
                    .set_viewport(0, [gui_resources.viewport.clone()])
                    .bind_pipeline_graphics(scene.pipelines[0].clone());

                //.bind_vertex_buffers(0, );
                //TODO: Need to create vertex buffer GPU side
            }
            _ => todo!(),
        }
    });
}
