#comment begin
// ######################################################################### //
// ### Vertex Shader ####################################################### //
#comment end

#type vertex
#version 430 core

// a_: attribute_
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texCoord;

// u_: uniform_
uniform mat4 u_projView;
uniform mat4 u_model;

// v_: varying
out vec2 v_tc;

void main() {
	v_tc = a_texCoord;
	gl_Position = u_projView * u_model * vec4(a_position, 1.0);
}


#comment begin
// ######################################################################### //
// ### Fragment Shader ##################################################### //
#comment end

#type fragment
#version 430 core

// o_: output_
layout(location = 0) out vec4 o_color;			

// u_: uniform
uniform sampler2D u_texture;

// v_: varying
in vec2 v_tc;

void main() {
	o_color = texture(u_texture, v_tc);
}