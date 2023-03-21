#compute
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0, rg32f) uniform image2D FT_T;
layout(binding = 1, rg32f) uniform image2D FT_I;

void main(void)
{
    ivec2 x = ivec2(gl_GlobalInvocationID.xy);
    vec2 T = imageLoad(FT_T, x).rg;
    vec2 I = imageLoad(FT_I, x).rg;
    float amplitude = sqrt(pow(T.r * I.r + T.g * I.g, 2) + pow(I.r * T.g - T.r * I.g, 2));
    float real = (T.r * I.r + T.g * I.r) / amplitude;
    float imag = (I.r * T.g - T.r * I.g) / amplitude;
    imageStore(FT_I, x, vec4(real, imag, 0.0, 0.0));
}