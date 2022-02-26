#include "LazyEngine/gepch.h"

#include "PhongMaterial.h"

namespace LazyEngine {


	// ##################################################################### //
	// ### PhongMaterial ################################################### //
	// ##################################################################### //

	Ref<Shader> PhongMaterial::s_shader = nullptr;
	std::string PhongMaterial::s_phongVertexShaderSrc = R"(
		#version 430 core

		// a_: attribute_
		layout(location = 0) in vec3 a_position;
		layout(location = 1) in vec3 a_normal;

		// u_: uniform_
		// --- these will be filled by the renderer --- //
		uniform mat4 u_projView;
		uniform mat4 u_worldToView;
		uniform mat4 u_viewToProjection;
		uniform mat3 u_normalMatrix;
		uniform mat4 u_model;

		uniform float u_zOffset;

		// --- these will be filled by the material --- //
		// FIXME: This should actually be part of the renderer, obviously
		uniform vec3 u_lightPosition;

		// v_: varying_
		// --- These are all in camera space --- //
		out vec3 v_position;
		out vec3 v_normal;
		out vec3 v_lightPosition;

		void main() {
			mat4 modelView = (u_worldToView * u_model);			

			// transform position to camera space (where the camera is at [0,0,0])
			vec4 positionCamSpace = modelView * vec4(a_position, 1);

			// apply the normal matrix (((modelView)^-1)^T) to the normal
			mat3 normalM = u_normalMatrix;
			v_normal = normalize(normalM * a_normal);
			
			// transform the light direction to camera space
			vec4 lightPosCam = (u_worldToView * vec4(u_lightPosition, 1.0));
			v_lightPosition = lightPosCam.xyz / lightPosCam.w;

			// set the position of this vertex in clip space
			gl_Position = u_viewToProjection * (positionCamSpace + vec4(0.f,0.f,u_zOffset,0.f));

			// dehomogenize the transformed camera space position
			v_position = positionCamSpace.xyz / positionCamSpace.w + vec3(0.f,0.f,u_zOffset);
		}
	)";

	std::string PhongMaterial::s_phongFragmentShaderSrc = R"(
		#version 430 core
		
		// o_: output_
		layout(location = 0) out vec4 o_color;

		// u_: uniform_
		uniform vec4 u_color;

		uniform float u_ambientFactor;
		uniform float u_diffuseFactor;
		uniform float u_specularFactor;
		uniform float u_shinyExponent;
		uniform vec4 u_lightColor;

		// Camera specifics
		uniform float u_near;
		uniform float u_far;

		// v_: varying_
		// --- These are all in camera space --- //
		in vec3 v_position;
		in vec3 v_normal;
		in vec3 v_lightPosition;
		
		float linearizeDepth(float depth, float zNear, float zFar) {
		    float z = 2.0 * depth - 1.0;
		    return 2.0 * zNear * zFar / (zFar + zNear - z * (zFar - zNear));
		}

 		void main() {
			vec3 normal = normalize(v_normal);
			vec3 toLight = normalize(v_lightPosition - v_position);
			vec3 toCamera = normalize(-v_position);
			vec3 reflection = 2.0 * dot(normal, toLight) * normal - toLight;
			
			vec3 ambientColor = u_ambientFactor * u_color.rgb;
			vec3 diffuseColor = u_diffuseFactor * u_color.rgb * clamp(dot(normal, toLight), 0.0, 1.0);
			float shiny = clamp(dot(toCamera, reflection), 0.0, 1.0);
			shiny = pow(shiny, u_shinyExponent);
			vec3 specularColor = u_specularFactor * u_lightColor.rgb * shiny;
			
			o_color = vec4(ambientColor + diffuseColor + specularColor, u_color.a);
		}
	)";

	PhongMaterial::PhongMaterial()
		: m_ambientFactor(0.2f)
		, m_diffuseFactor(0.9f)
		, m_specularFactor(0.6f)
		, m_specularExponent(5.f)
		, m_lightPosition(glm::vec3(10.f,10.f,3.f))
		, m_lightColor(glm::vec4(1.f))
		, m_color(glm::vec4(0.9f, 0.1f, 0.9f, 1.f))
		, m_zOffset(0.f)
	{
		initialize();
		updateUniforms();
	}

	void PhongMaterial::initialize() {
		if (s_shader == nullptr) {
			s_shader = Shader::create("Phong", s_phongVertexShaderSrc, s_phongFragmentShaderSrc);
		}
	}

	void PhongMaterial::updateUniforms() {
		updateUniformsForShader(PhongMaterial::s_shader);
	}

	void PhongMaterial::updateUniformsForShader(Ref<Shader> shader) {
		shader->bind();
		shader->uniformFloat("u_ambientFactor", m_ambientFactor);
		shader->uniformFloat("u_diffuseFactor", m_diffuseFactor);
		shader->uniformFloat("u_specularFactor", m_specularFactor);
		shader->uniformFloat("u_shinyExponent", m_specularExponent);
		shader->uniformFloat3("u_lightPosition", m_lightPosition);
		shader->uniformFloat4("u_lightColor", m_lightColor);
		shader->uniformFloat4("u_color", m_color);
		shader->uniformFloat("u_zOffset", m_zOffset);
		shader->unbind();
	}


	// ##################################################################### //
	// ### ColoredPhongMaterial ############################################ //
	// ##################################################################### //

	Ref<Shader> ColoredPhongMaterial::s_shader = nullptr;
	Ref<Shader> ColoredPhongMaterial::s_quadOutlineShader = nullptr;
	Ref<Shader> ColoredPhongMaterial::s_triangleOutlineShader = nullptr;
	Ref<Shader> ColoredPhongMaterial::s_pointShader = nullptr;

#if 0
	std::string ColoredPhongMaterial::s_coloredPhongVertexShaderForGeometrySrc = R"(
		#version 430 core

		// a_: attribute_
		layout(location = 0) in vec3 a_position;
		layout(location = 1) in vec3 a_normal;
		layout(location = 2) in vec4 a_color;

		out vec3 i_position;
		out vec3 i_normal;
		out vec4 i_color;

		void main() {
			i_position = a_position;
			i_normal = a_normal;
			i_color = a_color;
			gl_Position = vec4(a_position, 1.0);
		}
	)";
#endif

	std::string ColoredPhongMaterial::s_coloredPhongVertexShaderSrc = R"(
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
		uniform vec4 u_color;

		// v_: varying_
		// --- These are all in camera space --- //
		out vec3 v_position;
		out vec3 v_normal;
		out vec4 v_color;
		out vec3 v_lightPosition;

		void main() {
			mat4 modelView = (u_worldToView * u_model);			

			// transform position to camera space (where the camera is at [0,0,0])
			vec4 positionCamSpace = modelView * vec4(a_position, 1);

			// apply the normal matrix (((modelView)^-1)^T) to the normal
			mat3 normalM = u_normalMatrix;
			v_normal = normalize(normalM * a_normal);
			
			// transform the light direction to camera space
			vec4 lightPosCam = (u_worldToView * vec4(u_lightPosition, 1.0));
			v_lightPosition = lightPosCam.xyz / lightPosCam.w;

			// set the position of this vertex in clip space
			gl_Position = u_viewToProjection * positionCamSpace;

			// dehomogenize the transformed camera space position
			v_position = positionCamSpace.xyz / positionCamSpace.w;

			// output the incoming color as is
			v_color = vec4(a_color.xyz, u_color.a);
		}
	)";

	std::string ColoredPhongMaterial::s_coloredPhongFragmentShaderSrc = R"(
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
	)";

	std::string ColoredPhongMaterial::s_quadOutlineGeometryShaderSrc = R"(
		#version 430 core
		
		layout(triangles) in;
		layout(line_strip, max_vertices=3) out;
		
		in vec3[] v_positionGeometry;
		in vec3[] v_normalGeometry;
		in vec4[] v_colorGeometry;
		in vec3[] v_lightPositionGeometry;
		
		out vec3 v_position;
		out vec3 v_normal;
		out vec4 v_color;
		out vec3 v_lightPosition;
		
		uniform bool u_invertColors;

		void main() {
			// simply pass the values through
			for(int i = 0; i < 3; ++i) {
				gl_Position = gl_in[i].gl_Position - vec4(0,0,0.0001,0);
				v_position = v_positionGeometry[i];
				v_normal = v_normalGeometry[i];
				if(u_invertColors) {
					v_color = 1 - v_colorGeometry[i];
					v_color.a = 1;
				} else {
					v_color = v_colorGeometry[i];
				}
				v_lightPosition = v_lightPositionGeometry[i];
				EmitVertex();
			}
			
			EndPrimitive();
		}
	)";

	std::string ColoredPhongMaterial::s_triangleOutlineGeometryShaderSrc = R"(
		#version 430 core
		
		layout(triangles) in;
		layout(line_strip, max_vertices=4) out;
		
		in vec3 v_positionGeometry[3];
		in vec3 v_normalGeometry[3];
		in vec4 v_colorGeometry[3];
		in vec3 v_lightPositionGeometry[3];
		
		out vec3 v_position;
		out vec3 v_normal;
		out vec4 v_color;
		out vec3 v_lightPosition;
		
		uniform bool u_invertColors;
		
		void main() {
			// simply pass the values through
			for(int i = 0; i < 3; ++i) {
				const int index = i;
				gl_Position = gl_in[index].gl_Position - vec4(0,0,0.0001,0);
				v_position = v_positionGeometry[index];
				v_normal = v_normalGeometry[index];
				if(u_invertColors) {
					v_color = 1 - v_colorGeometry[i];
					v_color.a = 1;
				} else {
					v_color = v_colorGeometry[i];
				}
				v_lightPosition = v_lightPositionGeometry[index];
				EmitVertex();
			}
			// repeat the first vertex to close the triangle
			gl_Position = gl_in[0].gl_Position - vec4(0,0,0.0001,0);
			v_position = v_positionGeometry[0];
			v_normal = v_normalGeometry[0];
				if(u_invertColors) {
					v_color = 1 - v_colorGeometry[0];
					v_color.a = 1;
				} else {
					v_color = v_colorGeometry[0];
				}
			v_lightPosition = v_lightPositionGeometry[0];
			EmitVertex();
		
			EndPrimitive();
		}
	)";

	// FIXME: This is incredibly slow!
	std::string ColoredPhongMaterial::s_pointsGeometryShaderSrc = R"(
		#version 430 core
		
		layout(points) in;
		layout(triangle_strip, max_vertices=4) out;
		
		in vec3 v_positionGeometry[1];
		in vec3 v_normalGeometry[1];
		in vec4 v_colorGeometry[1];
		in vec3 v_lightPositionGeometry[1];
		
		out vec3 v_position;
		out vec3 v_normal;
		out vec4 v_color;
		out vec3 v_lightPosition;
		
		void main() {
			// create a billboard from the point
			vec4 offsets[4] = {
				vec4(-1,-1, 0, 0),
				vec4( 1,-1, 0, 0),
				vec4( 1, 1, 0, 0),
				vec4(-1, 1, 0, 0),
			};

			// simply pass the values through
			for(int i = 0; i < 4; ++i) {
				v_position = v_positionGeometry[0];
				v_normal = v_normalGeometry[0];
				v_color = v_colorGeometry[0];
				v_lightPosition = v_lightPositionGeometry[0];
				gl_Position = gl_in[0].gl_Position + (offsets[i]);
				EmitVertex();
			}
		
			EndPrimitive();
		}
	)";

	ColoredPhongMaterial::ColoredPhongMaterial()
		: PhongMaterial()
	{
		initialize();
		updateUniforms();
	}

	void ColoredPhongMaterial::initialize() {
		if (s_shader == nullptr) {
			s_shader = Shader::create("ColoredPhong", s_coloredPhongVertexShaderSrc, s_coloredPhongFragmentShaderSrc);
		}
		if (s_quadOutlineShader == nullptr) {
			initializeQuadOutlineShader();
		}
		if (s_triangleOutlineShader == nullptr) {
			initializeTriangleOutlineShader();
		}
		if (s_pointShader == nullptr) {
			initializePointShader();
		}
	}

	inline void replaceAll(std::string& input, const std::string& from, const std::string& to) {
		size_t startPosition = 0;
		while ((startPosition = input.find(from, startPosition)) != std::string::npos) {
			input.replace(startPosition, from.length(), to);
			startPosition += to.length();
		}
	}

	void ColoredPhongMaterial::initializeQuadOutlineShader() {
		// replace all outputs of the vertex shader source to match the inputs of the geometry shader
		std::string vertexSource = s_coloredPhongVertexShaderSrc;
		replaceAll(vertexSource, "v_position", "v_positionGeometry");
		replaceAll(vertexSource, "v_normal", "v_normalGeometry");
		replaceAll(vertexSource, "v_color", "v_colorGeometry");
		replaceAll(vertexSource, "v_lightPosition", "v_lightPositionGeometry");

		std::unordered_map<ShaderRoutine, std::string> sources;
		sources[ShaderRoutine::VertexShader] = vertexSource;
		sources[ShaderRoutine::GeometryShader] = s_quadOutlineGeometryShaderSrc;
		sources[ShaderRoutine::FragmentShader] = s_coloredPhongFragmentShaderSrc;
		s_quadOutlineShader = Shader::create("ColoredPhongQuadOutlines", sources);
	}
	
	void ColoredPhongMaterial::initializeTriangleOutlineShader() {
		// replace all outputs of the vertex shader source to match the inputs of the geometry shader
		std::string vertexSource = s_coloredPhongVertexShaderSrc;
		replaceAll(vertexSource, "v_position", "v_positionGeometry");
		replaceAll(vertexSource, "v_normal", "v_normalGeometry");
		replaceAll(vertexSource, "v_color", "v_colorGeometry");
		replaceAll(vertexSource, "v_lightPosition", "v_lightPositionGeometry");

		std::unordered_map<ShaderRoutine, std::string> sources;
		sources[ShaderRoutine::VertexShader] = vertexSource;
		sources[ShaderRoutine::GeometryShader] = s_triangleOutlineGeometryShaderSrc;
		sources[ShaderRoutine::FragmentShader] = s_coloredPhongFragmentShaderSrc;
		s_triangleOutlineShader = Shader::create("ColoredPhongTriangleOutlines", sources);
	}

	void ColoredPhongMaterial::initializePointShader() {
		// replace all outputs of the vertex shader source to match the inputs of the geometry shader
		std::string vertexSource = s_coloredPhongVertexShaderSrc;
		replaceAll(vertexSource, "v_position", "v_positionGeometry");
		replaceAll(vertexSource, "v_normal", "v_normalGeometry");
		replaceAll(vertexSource, "v_color", "v_colorGeometry");
		replaceAll(vertexSource, "v_lightPosition", "v_lightPositionGeometry");

		std::unordered_map<ShaderRoutine, std::string> sources;
		sources[ShaderRoutine::VertexShader] = vertexSource;
		sources[ShaderRoutine::GeometryShader] = s_pointsGeometryShaderSrc;
		sources[ShaderRoutine::FragmentShader] = s_coloredPhongFragmentShaderSrc;
		s_pointShader = Shader::create("ColoredPhongPoints", sources);
	}

	void ColoredPhongMaterial::updateUniforms() {
		updateUniformsForShader(ColoredPhongMaterial::s_shader);
		updateUniformsForShader(ColoredPhongMaterial::s_quadOutlineShader);
		updateUniformsForShader(ColoredPhongMaterial::s_triangleOutlineShader);
	}


	// ##################################################################### //
	// ### TexturedPhongMaterial ########################################### //
	// ##################################################################### //

	Ref<Shader> TexturedPhongMaterial::s_shader = nullptr;
	std::string TexturedPhongMaterial::s_texturedPhongVertexShaderSrc = R"(
		#version 430 core

		// a_: attribute_
		layout(location = 0) in vec3 a_position;
		layout(location = 1) in vec3 a_normal;
		layout(location = 2) in vec2 a_textureCoordinates;

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
		out vec3 v_position;
		out vec3 v_normal;
		out vec3 v_lightPosition;

		out vec2 v_textureCoordinates;

		void main() {
			mat4 modelView = (u_worldToView * u_model);			

			// transform position to camera space (where the camera is at [0,0,0])
			vec4 positionCamSpace = modelView * vec4(a_position, 1);

			// apply the normal matrix (((modelView)^-1)^T) to the normal
			v_normal = normalize(u_normalMatrix * a_normal);
			
			// transform the light direction to camera space
			vec4 lightPosCam = (u_worldToView * vec4(u_lightPosition, 1.0));
			v_lightPosition = lightPosCam.xyz / lightPosCam.w;

			// set the position of this vertex in clip space
			gl_Position = u_viewToProjection * positionCamSpace;

			// dehomogenize the transformed camera space position
			v_position = positionCamSpace.xyz / positionCamSpace.w;

			// output the texture coordinate for this vertex
			v_textureCoordinates = a_textureCoordinates;
		}
	)";

	std::string TexturedPhongMaterial::s_texturedPhongFragmentShaderSrc = R"(
		#version 430 core
		
		// o_: output_
		layout(location = 0) out vec4 o_color;

		// u_: uniform_
		uniform vec4 u_color;

		uniform float u_ambientFactor;
		uniform float u_diffuseFactor;
		uniform float u_specularFactor;
		uniform float u_shinyExponent;
		uniform vec4 u_lightColor;

		uniform sampler2D u_texture;
		uniform float u_colorTintFactor;

		// v_: varying_
		// --- These are all in camera space --- //
		in vec3 v_position;
		in vec3 v_normal;
		in vec3 v_lightPosition;
		
		in vec2 v_textureCoordinates;
		
		void main() {
			vec4 texColor = texture(u_texture, v_textureCoordinates);
			vec4 color = mix(texColor, u_color, u_colorTintFactor);

			vec3 normal = normalize(v_normal);
			vec3 toLight = normalize(v_lightPosition - v_position);
			vec3 toCamera = normalize(-v_position);
			vec3 reflection = 2.0 * dot(normal, toLight) * normal - toLight;
			
			vec3 ambientColor = u_ambientFactor * color.rgb;
			vec3 diffuseColor = u_diffuseFactor * color.rgb * clamp(dot(normal, toLight), 0.0, 1.0);
			float shiny = clamp(dot(toCamera, reflection), 0.0, 1.0);
			shiny = pow(shiny, u_shinyExponent);
			vec3 specularColor = u_specularFactor * u_lightColor.rgb * shiny;
			
			o_color = vec4(ambientColor + diffuseColor + specularColor, u_color.a);
		}
	)";

	TexturedPhongMaterial::TexturedPhongMaterial(Ref<Texture2D> texture, float colorTintFactor)
		: PhongMaterial()
		, m_texture(texture)
		, m_colorTintFactor(colorTintFactor)
	{
		initialize();
		updateUniforms();
	}

	void TexturedPhongMaterial::initialize() {
		if (s_shader == nullptr) {
			s_shader = Shader::create("TexturedPhong", s_texturedPhongVertexShaderSrc, s_texturedPhongFragmentShaderSrc);
		}
	}

	void TexturedPhongMaterial::updateUniforms() {
		auto shader = TexturedPhongMaterial::s_shader;
		updateUniformsForShader(shader);

		shader->bind();
		shader->uniformFloat("u_colorTintFactor", m_colorTintFactor);
		constexpr int textureSlot = 0;
		m_texture->bind(textureSlot);
		shader->uniformInt("u_texture", textureSlot);
		shader->unbind();
	}


	// ##################################################################### //
	// ### BlinnPhongMaterial ############################################## //
	// ##################################################################### //

	Ref<Shader> BlinnPhongMaterial::s_shader = nullptr;

	std::string BlinnPhongMaterial::s_blinnPhongFragmentShaderSrc = R"(
		#version 430 core
		
		// o_: output_
		layout(location = 0) out vec4 o_color;

		// u_: uniform_
		uniform vec4 u_color;

		uniform float u_ambientFactor;
		uniform float u_diffuseFactor;
		uniform float u_specularFactor;
		uniform float u_shinyExponent;
		uniform vec4 u_lightColor;

		// v_: varying_
		// --- These are all in camera space --- //
		in vec3 v_position;
		in vec3 v_normal;
		in vec3 v_lightPosition;
		
		in vec2 v_textureCoordinates;

		void main() {
			vec3 normal = normalize(v_normal);
			vec3 toLight = normalize(v_lightPosition - v_position);
			vec3 toCamera = normalize(-v_position);
			vec3 halfway = normalize(toLight + toCamera);
			
			vec3 ambientColor = u_ambientFactor * u_color.rgb;
			vec3 diffuseColor = u_diffuseFactor * u_color.rgb * clamp(dot(normal, toLight), 0.0, 1.0);
			float shiny = clamp(dot(normal, halfway), 0.0, 1.0);
			shiny = pow(shiny, u_shinyExponent);
			vec3 specularColor = u_specularFactor * u_lightColor.rgb * shiny;
			
			o_color = vec4(ambientColor + diffuseColor + specularColor, u_color.a);
		}
	)";

	BlinnPhongMaterial::BlinnPhongMaterial()
		: PhongMaterial()
	{
		// FIXME: this is not working yet
		initialize();
		updateUniforms();
	}

	void BlinnPhongMaterial::initialize() {
		if (s_shader == nullptr) {
			s_shader = Shader::create("BlinnPhong", s_phongVertexShaderSrc, s_blinnPhongFragmentShaderSrc);
		}
	}

	void BlinnPhongMaterial::updateUniforms() {
		updateUniformsForShader(BlinnPhongMaterial::s_shader);
	}


	// ##################################################################### //
	// ### TexturePhongMaterial ############################################ //
	// ##################################################################### //

	Ref<Shader> TexturedBlinnPhongMaterial::s_shader = nullptr;
	std::string TexturedBlinnPhongMaterial::s_texturedBlinnPhongFragmentShaderSrc = R"(
		#version 430 core
		
		// o_: output_
		layout(location = 0) out vec4 o_color;

		// u_: uniform_
		uniform vec4 u_color;

		uniform float u_ambientFactor;
		uniform float u_diffuseFactor;
		uniform float u_specularFactor;
		uniform float u_shinyExponent;
		uniform vec4 u_lightColor;

		uniform sampler2D u_texture;
		uniform float u_colorTintFactor;

		// v_: varying_
		// --- These are all in camera space --- //
		in vec3 v_position;
		in vec3 v_normal;
		in vec3 v_lightPosition;
		
		in vec2 v_textureCoordinates;
		
		void main() {
			vec4 texColor = texture(u_texture, v_textureCoordinates);
			vec4 color = mix(texColor, u_color, u_colorTintFactor);

			vec3 normal = normalize(v_normal);
			vec3 toLight = normalize(v_lightPosition - v_position);
			vec3 toCamera = normalize(-v_position);
			vec3 halfway = normalize(toLight + toCamera);
			
			vec3 ambientColor = u_ambientFactor * color.rgb;
			vec3 diffuseColor = u_diffuseFactor * color.rgb * clamp(dot(normal, toLight), 0.0, 1.0);
			float shiny = clamp(dot(normal, halfway), 0.0, 1.0);
			shiny = pow(shiny, u_shinyExponent);
			vec3 specularColor = u_specularFactor * u_lightColor.rgb * shiny;
			
			o_color = vec4(ambientColor + diffuseColor + specularColor, u_color.a);
		}
	)";

	TexturedBlinnPhongMaterial::TexturedBlinnPhongMaterial(Ref<Texture2D> texture, float colorTintFactor)
		: TexturedPhongMaterial(texture, colorTintFactor)
	{
		initialize();
		updateUniforms();
	}

	void TexturedBlinnPhongMaterial::initialize() {
		if (s_shader == nullptr) {
			s_shader = Shader::create("TexturedBlinnPhong", s_texturedPhongVertexShaderSrc, s_texturedBlinnPhongFragmentShaderSrc);
		}
	}

	void TexturedBlinnPhongMaterial::updateUniforms() {
		auto shader = TexturedBlinnPhongMaterial::s_shader;
		updateUniformsForShader(shader);

		shader->bind();
		shader->uniformFloat("u_colorTintFactor", m_colorTintFactor);
		constexpr int textureSlot = 0;
		m_texture->bind(textureSlot);
		shader->uniformInt("u_texture", textureSlot);
		shader->unbind();
	}
}