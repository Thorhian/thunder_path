#version 450

layout(location = 0) in vec4 v_color;

layout(location = 0) out vec4 f_color;

float near = 0.1;
float far = 20;

//https://learnopengl.com/Advanced-OpenGL/Depth-testing
float LinearizeDepth(float depth)
{
    float z = depth * 2.0 - 1.0; // back to NDC
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main() {
    vec4 newColor = vec4(0.0, 0.0, 1.0, 1.0);
    if(v_color.b > 0.90) {
        newColor = vec4(v_color.r +  sqrt(gl_FragCoord.z), v_color.gba);
    } else {
        newColor = v_color;
    }
    f_color = newColor;
}
