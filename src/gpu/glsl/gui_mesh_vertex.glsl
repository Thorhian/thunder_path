#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} mvp_ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position =
        mvp_ubo.proj * mvp_ubo.view * mvp_ubo.model * vec4(inPosition, 1.0);
    fragColor = inColor;
}
