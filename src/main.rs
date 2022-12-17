use nalgebra::Vector2;
pub mod compute_worker;

fn main() {
    let center1 = Vector2::new(40.0, 1.0);
    let center2 = Vector2::new(1.0, 2.0);
    let radius = 6.0 / 0.2;
    let quad = compute_worker::find_rectangle_points(center1, center2, radius);
    println!("Sorted Quad: {}", quad);
}
