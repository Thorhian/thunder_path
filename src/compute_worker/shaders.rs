pub mod target_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/compute_worker/glsl/target_vertex.glsl"
    }
}

pub mod target_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/compute_worker/glsl/target_frag.glsl"
    }
}

pub mod edge_detection {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/compute_worker/glsl/edge_detection.glsl"
    }
}

pub mod edge_expansion {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/compute_worker/glsl/edge_expand.glsl"
    }
}
