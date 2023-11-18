pub mod gpu;
pub mod job;

use clap::Parser;
use gpu::PipelineDependencies;
use nalgebra::Vector3;
use russimp::scene;
use russimp::scene::Scene;
use vulkano::DeviceSize;
use vulkano::buffer::{Buffer, BufferUsage, BufferCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage};
use vulkano::sync::GpuFuture;
use std::path::PathBuf;
use std::sync::Arc;

extern crate clap;
//use clap::{Arg, App, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    target_model: PathBuf,
    stock_model: PathBuf,
    tool_diameter: f32,
    step_down: f32,
    x_offset: f32,
    y_offset: f32,
    z_offset: f32,

    #[arg(short, long, default_value_t = true)]
    gui: bool,
}

fn main() {
    let cli = Cli::parse();
    let mut post_process_target = scene::PostProcessSteps::new();
    let mut post_process_stock = scene::PostProcessSteps::new();
    post_process_target.push(scene::PostProcess::ForceGenerateNormals);
    post_process_stock.push(scene::PostProcess::ForceGenerateNormals);

    let binding = cli.target_model.into_os_string();
    let target_path = binding.to_str().unwrap();
    let binding = cli.stock_model.into_os_string();
    let stock_path = binding.to_str().unwrap();
    let target_scene = Scene::from_file(target_path, post_process_target).unwrap();
    let stock_scene = Scene::from_file(stock_path, post_process_stock).unwrap();

    let target_mesh = target_scene.meshes.first().unwrap();
    let stock_mesh = stock_scene.meshes.first().unwrap();

    let new_job = job::Job {
        tool_diameter: cli.tool_diameter,
        step_down_height: cli.step_down,
        target_mesh,
        stock_mesh,
    };

    let result = gpu::GPUInstance::initialize_instance(true);
    let (gpu_instance, event_loop_result, gui_resources_result) = 
        result.expect("failed to initialize gpu");

    let gpu_instance = Arc::new(gpu_instance);

    if cli.gui {
        let event_loop = event_loop_result.expect("failed to create event loop");
        let gui_resources = gui_resources_result.expect("failed to create gui resources");
        
        let target_mesh_pipeline = gpu_instance.create_gui_mesh_pipeline(&gui_resources);
        let (target_mesh, bounds) = gpu::import_verts(new_job.target_mesh, Vector3::new(0.0, 0.0, 1.0));
        let target_mesh_model = gpu::MeshModel {
            vbo_contents: target_mesh,
            mesh_type: gpu::MeshType::TargetMesh,
            bounds
        };

        let pipelines = vec![target_mesh_pipeline];
        let mesh_models = vec![target_mesh_model];
        let mesh_pipe_indices = vec![0];

        let mut pipeline_deps: Vec<PipelineDependencies> = Vec::new();
        for model in mesh_models {
            let vbo_size = model.vbo_contents.len();
            let vbo_stage = Buffer::from_iter(
                &gpu_instance.standard_mem_alloc,
                vulkano::buffer::BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload,
                    ..Default::default()
                },
                model.vbo_contents.into_iter()
            ).unwrap();

            let vbo_device = Buffer::new_slice::<gpu::ModelVertex>(
                &gpu_instance.standard_mem_alloc,
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::DeviceOnly,
                    ..Default::default()
                },
                (std::mem::size_of::<gpu::ModelVertex>() * vbo_size) as DeviceSize
            ).unwrap();

            let queue_index = gpu_instance.queue_family_indices[0].clone().0;
            let gfx_queue = &gpu_instance.queues[queue_index as usize];

            let mut cbb = AutoCommandBufferBuilder::primary(
                &gpu_instance.command_buff_allocator,
                gfx_queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            ).unwrap();

            cbb.copy_buffer(CopyBufferInfo::buffers(
                vbo_stage,
                vbo_device.clone()
            )).unwrap();

            let cb = cbb.build().unwrap();
            cb.execute(gfx_queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();
            
            let dependency = PipelineDependencies {
                vbo: vbo_device.clone()
            };

            pipeline_deps.push(dependency);
        }

        let scene = gpu::SceneContents {
            pipeline_dependencies: pipeline_deps,
            pipelines,
            mesh_pipe_indices
        };

        gpu::window::run_gui_loop(
            gpu_instance.clone(), event_loop, gui_resources, scene
        );
    }

    //gpu::process_job(new_job);
}
