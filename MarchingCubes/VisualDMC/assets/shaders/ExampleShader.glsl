#comment begin
// ######################################################################### //
// ### Vertex Shader ####################################################### //
#comment end

#type vertex
#version 430 core

// a_: attribute_
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_textureCoordinates;

// u_: uniform_
uniform mat4 u_projView;
uniform mat4 u_model;

// v_: varying_
out vec4 v_position_WorldCoords;
out vec3 v_normal_WorldCoords;
out vec4 v_color;

void main() {
    v_position_WorldCoords = u_model * vec4(a_position, 1);
	v_normal_WorldCoords = (u_model * vec4(a_normal, 0)).xyz;
	//v_color = a_color;
	v_color = vec4(0.8,0.1,0.8,1.0);
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
uniform vec3 u_cameraPosition;

// v_: varying_
in vec4 v_position_WorldCoords;
in vec3 v_normal_WorldCoords;
in vec4 v_color;

void main() {
	vec3 lightDirection = vec3(-1,-1,-1);
	vec4 lightColor = vec4(0.9, 0.9, 0.9, 1.0);
	float ambient = 0.2;
	float diffuse = 0.6;
	float specular = 0.4;
	float shiny = 4;

	vec3 n = v_normal_WorldCoords;
	vec3 l = normalize(-lightDirection);
	vec3 v = normalize(u_cameraPosition - v_position_WorldCoords.xyz);
	vec3 r = 2.*dot(n, l)*n-l;
	vec3 ambient_color = ambient * v_color.rgb;
	vec3 diffuse_color = diffuse * v_color.rgb * clamp(dot(n,l), 0, 1);
	vec3 specular_color = specular * lightColor.rgb * pow(clamp(dot(v,r), 0, 1), shiny);
	o_color = vec4(ambient_color + diffuse_color + specular_color, v_color.a);
	//o_color = vec4((n*0.5)+0.5, 1.0);
}