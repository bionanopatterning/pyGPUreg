#compute
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0, rgba32f) uniform image2D FTs;

void main(void)
{
    ivec2 x = ivec2(gl_GlobalInvocationID.xy);
    vec4 val = imageLoad(FTs, x).rgba;
    float amplitude = sqrt(pow(val.r * val.b + val.g * val.a, 2) + pow(val.b * val.g - val.r * val.a, 2));
    float real = (val.r * val.b + val.g * val.a) / amplitude;
    float imag = (val.b * val.g - val.r * val.a) / amplitude;
    imageStore(FTs, x, vec4(real, imag, real, imag));
}