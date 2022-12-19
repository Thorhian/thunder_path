pub mod compute_worker;

use std::path::PathBuf;
use nalgebra::Vector2;
use clap::Parser;
use russimp::scene;
use russimp::scene::Scene;

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
    println!("{}", cli.target_model.as_os_str().to_str().unwrap());
    let center1 = Vector2::new(40.0, 1.0);
    let center2 = Vector2::new(1.0, 2.0);
    let radius = 6.0 / 0.2;
    let quad = compute_worker::find_rectangle_points(center1, center2, radius);
    println!("Sorted Quad: {}", quad);

    let mut post_process_target = scene::PostProcessSteps::new();
    let mut post_process_stock = scene::PostProcessSteps::new();
    post_process_target.push(scene::PostProcess::ForceGenerateNormals);
    post_process_stock.push(scene::PostProcess::ForceGenerateNormals);

    let binding = cli.target_model.into_os_string();
    let target_path = binding.to_str().unwrap();
    let binding = cli.stock_model.into_os_string();
    let stock_path = binding.to_str().unwrap();
    let _target_scene = Scene::from_file(target_path, post_process_target).unwrap();
    let stock_scene =  Scene::from_file(stock_path, post_process_stock).unwrap();

    println!("Stock Vertices: {:?}", stock_scene.meshes[0].vertices);

}
