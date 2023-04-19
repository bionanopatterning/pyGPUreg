#compute
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0) uniform sampler2D original_image;
layout(binding = 1, r32f) uniform image2D shifted_image;

uniform float dx;
uniform float dy;
uniform int W;
uniform int H;
uniform int edge_mode;

void main(void)
{
    ivec2 x = ivec2(gl_GlobalInvocationID.xy);
    vec2 sample_x = (x + vec2(dx, dy)) / vec2(W, H);
    if (edge_mode > 0)
    {
        imageStore(shifted_image, x, texture(original_image, sample_x));
    }
    else  // otherwise, edge_mode = 0 = zero outside of original image.
    {
        if ((sample_x.x < 1) && (sample_x.y < 1) && (sample_x.x > 0) && (sample_x.y > 0))
        {
            imageStore(shifted_image, x, texture(original_image, sample_x));
        }
    }
}