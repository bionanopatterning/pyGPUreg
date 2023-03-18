#compute
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0, r32f) uniform image2D original_image;
layout(binding = 1, r32f) uniform image2D shifted_image;

uniform int dx;
uniform int dy;
uniform int N;
uniform int edge_mode;

void main(void)
{
    ivec2 x = ivec2(gl_GlobalInvocationID.xy);
    ivec2 sample_x = x + ivec2(dx, dy);
    if ((sample_x.x < N) && (sample_x.y < N) && (sample_x.x > 0) && (sample_x.y > 0))
    {
        imageStore(shifted_image, x, imageLoad(original_image, sample_x));
    }
    else
    {
        if (edge_mode == 0) // zero
        {
            //imageStore(shifted_image, x, int(0));
        }
        else if (edge_mode == 1) // repeat
        {
            // TODO
        }
        else if (edge_mode == 2) // reflect
        {
            // TODO
        }
    }
}