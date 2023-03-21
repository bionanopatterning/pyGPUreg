#compute
#version 430 core
#define M_PI 3.1415926535897932384626433832795

layout (local_size_x = 16, local_size_y = 16) in;
layout (binding = 0, r32f) uniform readonly image2D tex_r;
layout (binding = 1, rg32f) uniform writeonly image2D tex_rg;

void main(void)
{
    ivec2 x = ivec2(gl_GlobalInvocationID.xy);
    imageStore(tex_rg, x, vec4(imageLoad(tex_r, x).r, 0.0, 0.0, 0.0));
}
