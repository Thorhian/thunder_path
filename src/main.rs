pub mod gpu;
pub mod job;

use clap::Parser;
use russimp::scene;
use russimp::scene::Scene;
use std::path::PathBuf;

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

    //gpu::process_job(new_job);
}
