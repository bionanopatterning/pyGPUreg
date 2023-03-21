#compute
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0, rgba32f) writeonly uniform image2D displacement;
layout(binding = 1, rgba32f) readonly uniform image2D pingpong0;
layout(binding = 2, rgba32f) readonly uniform image2D pingpong1;

uniform int pingpong;
uniform float N;

void main(void)
{
	ivec2 x = ivec2(gl_GlobalInvocationID.xy);
	float perms[] = {1.0, -1.0};
	int index = int(mod((int(x.x + x.y)), 2));
	float perm = perms[index];
	if(pingpong == 0)
	{
		vec4 hn = perm / N * imageLoad(pingpong0, x);
		imageStore(displacement, x, vec4(hn));
	}
	else if(pingpong == 1)
	{
		vec4 hn = perm / N * imageLoad(pingpong1, x);
		imageStore(displacement, x, vec4(hn));
	}
}

