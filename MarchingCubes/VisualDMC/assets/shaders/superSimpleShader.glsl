#comment begin
// ######################################################################### //
// ### Vertex Shader ####################################################### //
#comment end

#type vertex
#version 430 core

// a_: attribute_
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_color;

// u_: uniform_
uniform mat4 u_projView;
uniform mat4 u_model;

// v_: varying_
out vec4 v_color;

void main() {
	v_color = a_color;
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

// u_: uniform_

// v_: varying_
in vec4 v_color;

void main() {
	o_color = v_color;
}