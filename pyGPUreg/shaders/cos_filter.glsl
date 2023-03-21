#compute
#version 430 core

layout (local_size_x = 16, local_size_y = 16) in;
layout (binding = 0, r32f) readonly uniform image2D cos_mask;
layout (binding = 1, rgba32f) uniform image2D data;

void main(void)
{
    ivec2 x = ivec2(gl_GlobalInvocationID.xy);
    vec4 val = imageLoad(data, x).rgba;
    float mask = imageLoad(cos_mask, x).r;
    imageStore(data, x, mask * val);
}