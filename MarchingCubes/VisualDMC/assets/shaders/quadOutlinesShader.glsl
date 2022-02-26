#comment begin
// ######################################################################### //
// ### Vertex Shader ####################################################### //
#comment end

#type vertex
#version 430 core

// a_: attribute_
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 a_color;

// u_: uniform_
// --- these will be filled by the renderer --- //
uniform mat4 u_projView;
uniform mat4 u_worldToView;
uniform mat4 u_viewToProjection;
uniform mat3 u_normalMatrix;
uniform mat4 u_model;

// --- these will be filled by the material --- //
// FIXME: This should actually be part of the renderer, obviously
uniform vec3 u_lightPosition;

// v_: varying_
// --- These are all in camera space --- //
out vec3 v_positionGeometry;
out vec3 v_normalGeometry;
out vec4 v_colorGeometry;
out vec3 v_lightPositionGeometry;

void main() {
	mat4 modelView = (u_worldToView * u_model);		
	// transform position to camera space (where the camera is at [0,0,0])
	vec4 positionCamSpace = modelView * vec4(a_position, 1);
	// apply the normal matrix (((modelView)^-1)^T) to the normal
	mat3 normalM = u_normalMatrix;
	v_normalGeometry = normalize(normalM * a_normal);
	
	// transform the light direction to camera space
	vec4 lightPosCam = (u_worldToView * vec4(u_lightPosition, 1.0));
	v_lightPositionGeometry = lightPosCam.xyz / lightPosCam.w
	// set the position of this vertex in clip space
	gl_Position = u_viewToProjection * positionCamSpace
	// dehomogenize the transformed camera space position
	v_positionGeometry = positionCamSpace.xyz / positionCamSpace.w
	// output the incoming color as is
	v_colorGeometry = a_color;
}

#comment begin
// ######################################################################### //
// ### Geometry Shader ##################################################### //
#comment end

#type geometry
#version 430 core

layout(triangles) in;
layout(line_strip, max_vertices=3) out;

in vec3 v_positionGeometry[];
in vec3 v_normalGeometry[];
in vec4 v_colorGeometry[];
in vec3 v_lightPositionGeometry[];

out vec3 v_position[];
out vec3 v_normal[];
out vec4 v_color[];
out vec3 v_lightPosition[];

void main() {
	// simply pass the values through
	for(int i = 0; i < 3; ++i) {
		gl_Position = gl_in[i].gl_Position;
		v_position[i] = v_positionGeometry[i];
		v_normal[i] = v_normalGeometry[i];
		v_color[i] = v_colorGeometry[i];
		v_lightPosition[i] = v_lightPositionGeometry[i];
		EmitVertex();
	}

	EndPrimitive();
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
uniform float u_ambientFactor;
uniform float u_diffuseFactor;
uniform float u_specularFactor;
uniform float u_shinyExponent;
uniform vec4 u_lightColor;

// v_: varying_
// --- These are all in camera space --- //
in vec3 v_position;
in vec3 v_normal;
in vec4 v_color;
in vec3 v_lightPosition;

void main() {
	vec3 normal = normalize(v_normal);
	vec3 toLight = normalize(v_lightPosition - v_position);
	vec3 toCamera = normalize(-v_position);
	vec3 reflection = 2.0 * dot(normal, toLight) * normal - toLight;
	
	vec3 ambientColor = u_ambientFactor * v_color.rgb;
	vec3 diffuseColor = u_diffuseFactor * v_color.rgb * clamp(dot(normal, toLight), 0.0, 1.0);
	float shiny = clamp(dot(toCamera, reflection), 0.0, 1.0);
	shiny = pow(shiny, u_shinyExponent);
	vec3 specularColor = u_specularFactor * u_lightColor.rgb * shiny;
	
	o_color = vec4(ambientColor + diffuseColor + specularColor, v_color.a);
}