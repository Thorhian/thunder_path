use russimp::mesh::Mesh;

pub struct Job<'a> {
    pub tool_diameter: f32,
    pub step_down_height: f32,
    pub target_mesh: &'a Mesh,
    pub stock_mesh: &'a Mesh,
}

