#compute
#version 430 core

layout (local_size_x = 16, local_size_y = 16) in;
layout (binding = 0, r32f) readonly uniform image2D cos_mask;
layout (binding = 1, rg32f) uniform image2D data;

void main(void)
{
    ivec2 x = ivec2(gl_GlobalInvocationID.xy);
    vec2 val = imageLoad(data, x).rg;
    float mask = imageLoad(cos_mask, x).r;
    imageStore(data, x, vec4(mask * val, 0.0, 0.0));
}