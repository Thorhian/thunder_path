use std::sync::Arc;

use vulkano::{
    instance::{Instance, InstanceCreateInfo},
    VulkanLibrary,
};

use vulkano::device::Queue;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};

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
