pub mod compute_worker;

use std::path::PathBuf;
use nalgebra::Vector2;
use clap::{Parser, Subcommand};

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
}
