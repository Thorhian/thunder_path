pub mod target_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/gpu/glsl/target_vertex.glsl"
    }
}

pub mod target_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/gpu/glsl/target_frag.glsl"
    }
}

pub mod edge_detection {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/gpu/glsl/edge_detection.glsl"
    }
}

pub mod edge_expansion {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/gpu/glsl/edge_expand.glsl"
    }
}

pub mod gui_mesh_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/gpu/glsl/gui_mesh_vertex.glsl"
    }
}
