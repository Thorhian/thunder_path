use std;
use nalgebra::Matrix2x4;
use nalgebra::Vector2;

fn find_rectangle_points(
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

    Matrix2x4::from_columns(&[point1, point2, point3, point4])
}

fn sort_rectangle_points(points: Matrix2x4<f32>) {
    let mut avg_center: Vector2<f32> = Vector2::zeros();
    let mut polar_coords: Vec<f32> = Vec::new();

    for col in points.column_iter() {
        avg_center += col;
        let mut polar_rot = col[1].atan2(col[0]);
        if polar_rot < 0.0  {
            polar_rot += std::f32::consts::PI * 2.0;
        }
        polar_coords.push(polar_rot);
    }

    avg_center = avg_center / 4.0;
    println!("Average Center: {}", avg_center);
    

    
}

fn main() {
    let center1 = Vector2::new(40.0, 1.0);
    let center2 = Vector2::new(1.0, 2.0);
    let radius = 6.0 / 0.2;
    let quad = find_rectangle_points(center1, center2, radius);
    sort_rectangle_points(quad);
}
