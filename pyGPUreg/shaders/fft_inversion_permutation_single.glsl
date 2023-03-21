#compute
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0, rg32f) uniform image2D ft_texture;

uniform int pingpong;
uniform float N;

void main(void)
{
	ivec2 x = ivec2(gl_GlobalInvocationID.xy);
	float perms[] = {1.0, -1.0};
	int index = int(mod((int(x.x + x.y)), 2));
	float perm = perms[index];
	vec2 hn = perm / N * imageLoad(ft_texture, x).rg;
	imageStore(ft_texture, x, vec4(hn, 0.0, 0.0));
}

