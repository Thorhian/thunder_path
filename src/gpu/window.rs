use std::sync::Arc;

use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferUsage, RenderPassBeginInfo, RenderingAttachmentInfo,
        RenderingInfo, SubpassContents,
    },
    device::QueueFlags,
    render_pass,
    swapchain::{acquire_next_image, SwapchainCreateInfo},
    sync::{self, GpuFuture},
    Validated, VulkanError,
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
    let queue_index = gpu_instance
        .physical_device
        .queue_family_properties()
        .iter()
        .position(|fam| fam.queue_flags.contains(QueueFlags::GRAPHICS))
        .unwrap();

    let graphics_queue = gpu_instance.queues[queue_index as usize].clone();

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
                    let (new_swapchain, new_images) = gui_resources
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: dimensions.into(),
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
                        );

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

                builder
                    .begin_rendering(RenderingInfo {
                        color_attachments: vec![Some(
                            RenderingAttachmentInfo {
                                load_op: render_pass::AttachmentLoadOp::Clear,
                                store_op: render_pass::AttachmentStoreOp::Store,
                                clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                                // Get the current frame imageview
                                ..RenderingAttachmentInfo::image_view(
                                    gui_resources.gui_framebuffers
                                        [image_index as usize]
                                        .attachments()
                                        .first()
                                        .unwrap()
                                        .clone(),
                                )
                            },
                        )],
                        ..Default::default()
                    })
                    .unwrap()
                    .set_viewport(
                        0,
                        [gui_resources.viewport.clone()].into_iter().collect(),
                    )
                    .unwrap();
                    //.bind_pipeline_graphics()
                //.set_viewport(0, [gui_resources.viewport.clone()])
                //.bind_pipeline_graphics(scene.pipelines[0].clone());

                //.bind_vertex_buffers(0, );
                //TODO: Need to create vertex buffer GPU side
            }
            _ => todo!(),
        }
    });
}
