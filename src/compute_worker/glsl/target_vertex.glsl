#version 450

layout(location = 0) in vec3 in_vert;
layout(location = 1) in vec4 in_color;

layout(location = 2) out vec4 v_color;

layout(binding = 0) uniform MatrixBlock {
    mat4 projection_matrix;
} matrices;


void main() {
  v_color = in_color;
  gl_Position = matrices.projection_matrix * vec4(in_vert, 1.0);
}
