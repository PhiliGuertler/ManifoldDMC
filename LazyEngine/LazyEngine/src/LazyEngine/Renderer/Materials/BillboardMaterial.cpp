#include "BillboardMaterial.h"

namespace LazyEngine {

	// ##################################################################### //
	// ### BillboardMaterial ############################################### //
	// ##################################################################### //

	BillboardMaterial::BillboardMaterial()
		: PhongMaterial()
		, m_worldSpaceSize(glm::vec2(0.05f))
		, m_screenSpaceSize(glm::vec2(0.05f))
		, m_useScreenSpaceSize(false)
		, m_color(glm::vec4(1.f, 1.f, 1.f, 1.f))
		, m_colorTintFactor(0.25f)
	{
		setAmbientFactor(1.f);
		setDiffuseFactor(1.f);
		setSpecularFactor(0.0f);
		setSpecularExponent(1);
	}

	BillboardMaterial::~BillboardMaterial() {
		// empty
	}

	Ref<Shader> BillboardMaterial::getShader() const {
		if (m_useScreenSpaceSize) {
			return getScreenSpaceBillboardShader();
		}
		else {
			return getWorldSpaceBillboardShader();
		}
	}

	void BillboardMaterial::updateUniforms() {
		auto worldSpaceShader = getWorldSpaceBillboardShader();
		updateUniformsForShader(worldSpaceShader);
		worldSpaceShader->bind();
		worldSpaceShader->uniformFloat2("u_size", m_worldSpaceSize);
		worldSpaceShader->uniformFloat4("u_color", m_color);
		worldSpaceShader->uniformFloat("u_colorTintFactor", m_colorTintFactor);
		
		auto screenSpaceShader = getScreenSpaceBillboardShader();
		updateUniformsForShader(screenSpaceShader);
		screenSpaceShader->bind();
		screenSpaceShader->uniformFloat2("u_size", m_screenSpaceSize);
		screenSpaceShader->uniformFloat4("u_color", m_color);
		screenSpaceShader->uniformFloat("u_colorTintFactor", m_colorTintFactor);
	}

	void BillboardMaterial::setWorldSpaceWidth(float width) {
		m_worldSpaceSize.x = width;
	}

	void BillboardMaterial::setWorldSpaceHeight(float height) {
		m_worldSpaceSize.y = height;
	}

	void BillboardMaterial::setWorldSpaceSize(const glm::vec2& size) {
		m_worldSpaceSize = size;
	}

	void BillboardMaterial::setScreenSpaceWidth(float width) {
		m_screenSpaceSize.x = width;
	}

	void BillboardMaterial::setScreenSpaceHeight(float height) {
		m_screenSpaceSize.y = height;
	}

	void BillboardMaterial::setScreenSpaceSize(const glm::vec2& size) {
		m_screenSpaceSize = size;
	}

	void BillboardMaterial::setUseScreenSpaceSize(bool value) {
		m_useScreenSpaceSize = value;
	}

	void BillboardMaterial::setColor(const glm::vec4& color) {
		m_color = color;
	}

	void BillboardMaterial::setColorTintFactor(float factor) {
		m_colorTintFactor = factor;
	}

	std::string BillboardMaterial::s_billboardVertexWorldSpaceSrc = R"(
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

		uniform vec2 u_size;

		// v_: varying_
		// --- These are all in camera space --- //
		out vec3 v_positionGeometry;
		out vec3 v_normalGeometry;
		out vec4 v_colorGeometry;
		out vec3 v_lightPositionGeometry;

		out vec3 v_right;
		out vec3 v_top;

		void main() {
			mat4 modelView = (u_worldToView * u_model);			

			// transform position to camera space (where the camera is at [0,0,0])
			vec4 positionCamSpace = modelView * vec4(a_position, 1);

			// apply the normal matrix (((modelView)^-1)^T) to the normal
			mat3 normalM = u_normalMatrix;
			v_normalGeometry = normalize(normalM * a_normal);
			
			// transform the light direction to camera space
			vec4 lightPosCam = (u_worldToView * vec4(u_lightPosition, 1.0));
			v_lightPositionGeometry = lightPosCam.xyz / lightPosCam.w;

			// set the position of this vertex in clip space
			gl_Position = u_viewToProjection * positionCamSpace;

			// dehomogenize the transformed camera space position
			v_positionGeometry = positionCamSpace.xyz / positionCamSpace.w;

			// output the incoming color as is
			v_colorGeometry = a_color;
			// Keep right and top in world space
			v_right = vec3(1,0,0) * u_size.x;
			v_top = vec3(0,1,0) * u_size.y;
		}
	)";

	std::string BillboardMaterial::s_billboardVertexScreenSpaceSrc = R"(
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
			v_lightPositionGeometry = lightPosCam.xyz / lightPosCam.w;

			// set the position of this vertex in clip space
			gl_Position = u_viewToProjection * positionCamSpace;

			// dehomogenize the transformed camera space position
			v_positionGeometry = positionCamSpace.xyz / positionCamSpace.w;

			// output the incoming color as is
			v_colorGeometry = a_color;
		}
	)";

	std::string BillboardMaterial::s_billboardGeometryScreenSpaceSrc = R"(
		#version 430 core

		layout(points) in;
		layout(triangle_strip, max_vertices=4) out;

		uniform mat4 u_worldToView;
		uniform mat4 u_viewToProjection;
		uniform vec2 u_size;

		in vec3[] v_positionGeometry;
		in vec3[] v_normalGeometry;
		in vec4[] v_colorGeometry;
		in vec3[] v_lightPositionGeometry;
		
		out vec3 v_position;
		out vec3 v_normal;
		out vec4 v_color;
		out vec3 v_lightPosition;
		out vec2 v_texCoords;

		void main() {
			// create a quad at the point's position
			vec4 center = gl_in[0].gl_Position;
			vec3 front = vec3(0,0,-1.0);

			vec4 offsets[4] = {
				vec4(-u_size.x, -u_size.y, 0.0, 1.0),
				vec4( u_size.x, -u_size.y, 0.0, 1.0),
				vec4(-u_size.x,  u_size.y, 0.0, 1.0),
				vec4( u_size.x,  u_size.y, 0.0, 1.0),
			};

			vec2 texCoords[4] = {
				vec2(0,0),
				vec2(0,1),
				vec2(1,0),
				vec2(1,1),
			};

			mat4 viewToWorld = inverse(u_worldToView);

			for (int i = 0; i < 4; ++i) {
				gl_Position = center / center.w + vec4(offsets[i].xy, 0, 0);
				v_position = v_positionGeometry[0] + offsets[i].xyz;
				vec4 normal = normalize(vec4(normalize(offsets[i].xyz) + front, 0));
				v_normal =  vec3(viewToWorld * normal);
				v_color = v_colorGeometry[0];
				v_lightPosition = v_lightPositionGeometry[0];
				v_texCoords = texCoords[i];
				EmitVertex();
			}

			EndPrimitive();
		}
	)";

	std::string BillboardMaterial::s_billboardGeometryWorldSpaceSrc = R"(
		#version 430 core

		layout(points) in;
		layout(triangle_strip, max_vertices=4) out;

		uniform mat4 u_worldToView;

		in vec3[] v_positionGeometry;
		in vec3[] v_normalGeometry;
		in vec4[] v_colorGeometry;
		in vec3[] v_lightPositionGeometry;
		
		in vec3[] v_right;
		in vec3[] v_top;

		out vec3 v_position;
		out vec3 v_normal;
		out vec4 v_color;
		out vec3 v_lightPosition;
		out vec2 v_texCoords;

		void main() {
			// create a quad at the point's position
			vec4 center = gl_in[0].gl_Position;
			vec3 front = vec3(0,0,-1.0);
			
			vec4 offsets[4] = {
				vec4(-v_right[0] - v_top[0], 0),
				vec4( v_right[0] - v_top[0], 0),
				vec4(-v_right[0] + v_top[0], 0),
				vec4( v_right[0] + v_top[0], 0),
			};

			vec2 texCoords[4] = {
				vec2(0,0),
				vec2(0,1),
				vec2(1,0),
				vec2(1,1),
			};

			mat4 viewToWorld = inverse(u_worldToView);

			for (int i = 0; i < 4; ++i) {
				gl_Position = center + offsets[i];
				v_position = v_positionGeometry[0] + offsets[i].xyz;
				vec4 normal = normalize(vec4(offsets[i].xyz + front, 0));
				v_normal =  vec3(viewToWorld * normal);
				v_color = v_colorGeometry[0];
				v_lightPosition = v_lightPositionGeometry[0];
				v_texCoords = texCoords[i];
				EmitVertex();
			}

			EndPrimitive();
		}
	)";


	// ##################################################################### //
	// ### ColoredBillboardMaterial ######################################## //
	// ##################################################################### //

	Ref<Shader> ColoredSphereBillboardMaterial::s_screenSpaceShader = nullptr;
	Ref<Shader> ColoredSphereBillboardMaterial::s_worldSpaceShader = nullptr;

	ColoredSphereBillboardMaterial::ColoredSphereBillboardMaterial()
		: BillboardMaterial()
	{
		initialize();
	}

	void ColoredSphereBillboardMaterial::initialize() {
		if (s_screenSpaceShader == nullptr) {
			// initialize screen space shader
			std::unordered_map<ShaderRoutine, std::string> sources;
			sources[ShaderRoutine::VertexShader] = BillboardMaterial::s_billboardVertexScreenSpaceSrc;
			sources[ShaderRoutine::GeometryShader] = BillboardMaterial::s_billboardGeometryScreenSpaceSrc;
			sources[ShaderRoutine::FragmentShader] = s_coloredSphereBillboardFragmentSrc;
			s_screenSpaceShader = Shader::create("ColoredBillboardSS", sources);
		}
		if (s_worldSpaceShader == nullptr) {
			// initialize world space shader
			std::unordered_map<ShaderRoutine, std::string> sources;
			sources[ShaderRoutine::VertexShader] = BillboardMaterial::s_billboardVertexWorldSpaceSrc;
			sources[ShaderRoutine::GeometryShader] = BillboardMaterial::s_billboardGeometryWorldSpaceSrc;
			sources[ShaderRoutine::FragmentShader] = s_coloredSphereBillboardFragmentSrc;
			s_worldSpaceShader = Shader::create("ColoredBillboardWS", sources);
		}
	}

	Ref<Shader> ColoredSphereBillboardMaterial::getScreenSpaceBillboardShader() const {
		return s_screenSpaceShader;
	}

	Ref<Shader> ColoredSphereBillboardMaterial::getWorldSpaceBillboardShader() const {
		return s_worldSpaceShader;
	}

	std::string ColoredSphereBillboardMaterial::s_coloredSphereBillboardFragmentSrc = R"(
		#version 430 core
		
		// o_: output_
		layout(location = 0) out vec4 o_color;

		// u_: uniform_
		uniform vec4 u_color;
		uniform float u_colorTintFactor;

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
		
		in vec2 v_texCoords;
		
		void main() {
			vec4 color = mix(v_color, u_color, u_colorTintFactor);

			// only colorize a sphere, discard the corners of the billboard quad
			if (length(v_texCoords * 2.0 - 1.0) > 1.0) discard;

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


	// ##################################################################### //
	// ### TexturedBillboardMaterial ####################################### //
	// ##################################################################### //

	Ref<Shader> TexturedBillboardMaterial::s_screenSpaceShader = nullptr;
	Ref<Shader> TexturedBillboardMaterial::s_worldSpaceShader = nullptr;

	TexturedBillboardMaterial::TexturedBillboardMaterial(const Ref<Texture2D>& texture)
		: BillboardMaterial()
		, m_texture(texture)
		, m_tintFactor(0.f)
	{
		initialize();
	}

	void TexturedBillboardMaterial::setTexture(const Ref<Texture2D>& texture) {
		m_texture = texture;
	}

	void TexturedBillboardMaterial::updateUniforms() {
		BillboardMaterial::updateUniforms();
		
		constexpr int textureSlot = 0;
		
		auto screenSpaceShader = getScreenSpaceBillboardShader();
		screenSpaceShader->bind();
		m_texture->bind(textureSlot);
		screenSpaceShader->uniformInt("u_texture", textureSlot);

		auto worldSpaceShader = getWorldSpaceBillboardShader();
		worldSpaceShader->bind();
		m_texture->bind(textureSlot);
		worldSpaceShader->uniformInt("u_texture", textureSlot);
	}


	Ref<Shader> TexturedBillboardMaterial::getScreenSpaceBillboardShader() const {
		return s_screenSpaceShader;
	}

	Ref<Shader> TexturedBillboardMaterial::getWorldSpaceBillboardShader() const {
		return s_worldSpaceShader;
	}

	void TexturedBillboardMaterial::initialize() {
		if (s_screenSpaceShader == nullptr) {
			// initialize screen space shader
			std::unordered_map<ShaderRoutine, std::string> sources;
			sources[ShaderRoutine::VertexShader] = BillboardMaterial::s_billboardVertexScreenSpaceSrc;
			sources[ShaderRoutine::GeometryShader] = BillboardMaterial::s_billboardGeometryScreenSpaceSrc;
			sources[ShaderRoutine::FragmentShader] = s_texturedBillboardFragmentSrc;
			s_screenSpaceShader = Shader::create("TexturedBillboardSS", sources);
		}
		if (s_worldSpaceShader == nullptr) {
			// initialize world space shader
			std::unordered_map<ShaderRoutine, std::string> sources;
			sources[ShaderRoutine::VertexShader] = BillboardMaterial::s_billboardVertexWorldSpaceSrc;
			sources[ShaderRoutine::GeometryShader] = BillboardMaterial::s_billboardGeometryWorldSpaceSrc;
			sources[ShaderRoutine::FragmentShader] = s_texturedBillboardFragmentSrc;
			s_worldSpaceShader = Shader::create("TexturedBillboardWS", sources);
		}
	}

	std::string TexturedBillboardMaterial::s_texturedBillboardFragmentSrc = R"(
		#version 430 core
		
		// o_: output_
		layout(location = 0) out vec4 o_color;

		// u_: uniform_
		uniform vec4 u_color;
		uniform float u_colorTintFactor;

		uniform float u_ambientFactor;
		uniform float u_diffuseFactor;
		uniform float u_specularFactor;
		uniform float u_shinyExponent;
		uniform vec4 u_lightColor;

		uniform sampler2D u_texture;

		// v_: varying_
		// --- These are all in camera space --- //
		in vec3 v_position;
		in vec3 v_normal;
		in vec4 v_color;
		in vec3 v_lightPosition;
		
		in vec2 v_texCoords;
		
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


	// ##################################################################### //
	// ### LineBillboardMaterial ########################################### //
	// ##################################################################### //

	LineBillboardMaterial::LineBillboardMaterial()
		: BillboardMaterial()
		, m_lineWidth(0.4f)
	{
		// empty
	}

	void LineBillboardMaterial::updateUniforms() {
		BillboardMaterial::updateUniforms();

		auto screenSpaceShader = getScreenSpaceBillboardShader();
		screenSpaceShader->bind();
		screenSpaceShader->uniformFloat("u_lineWidth", m_lineWidth);

		auto worldSpaceShader = getWorldSpaceBillboardShader();
		worldSpaceShader->bind();
		worldSpaceShader->uniformFloat("u_lineWidth", m_lineWidth);
	}

	std::string LineBillboardMaterial::s_lineBillboardGeometryShaderScreenSpace = R"(
		#version 430 core
		
		layout(lines) in;
		layout(triangle_strip, max_vertices=4) out;

		uniform mat4 u_worldToView;
		uniform mat4 u_viewToProjection;
		uniform float u_lineWidth;
		
		in vec3[] v_positionGeometry;
		in vec3[] v_normalGeometry;
		in vec4[] v_colorGeometry;
		in vec3[] v_lightPositionGeometry;

		out vec3 v_position;
		out vec3 v_normal;
		out vec4 v_color;
		out vec3 v_lightPosition;
		out vec2 v_texCoords;

		void main() {
			// create a quad that is equivalent to the line, but a quad
			vec4 begin = gl_in[0].gl_Position;
			vec4 end = gl_in[1].gl_Position;
			
			float offsetFactors[4] = {
				-u_lineWidth * 0.5f,
				u_lineWidth * 0.5f,
				-u_lineWidth * 0.5f,
				u_lineWidth * 0.5f,
			};

			vec2 texCoords[4] = {
				vec2(0,0),
				vec2(0,1),
				vec2(1,0),
				vec2(1,1),
			};

			vec4 origins[4] = {
				begin, begin, end, end,
			};

			vec3 positions[4] = {
				v_positionGeometry[0],
				v_positionGeometry[0],
				v_positionGeometry[1],
				v_positionGeometry[1],
			};

			vec3 normals[4] = {
				v_normalGeometry[0],
				v_normalGeometry[0],
				v_normalGeometry[1],
				v_normalGeometry[1],
			};

			vec4 colors[4] = {
				v_colorGeometry[0],
				v_colorGeometry[0],
				v_colorGeometry[1],
				v_colorGeometry[1],
			};

			for (int i = 0; i < 4; ++i) {
				v_position = positions[i] + offsets[i].xyz;
				vec3 toCamera = normalize(-v_position);
				vec3 lineDirection = normalize(v_positionGeometry[1] - v_positionGeometry[0]);
				vec3 right = cross(toCamera, lineDirection);
				gl_Position = origins[i]/origins[i].w + vec4(right * offsetFactors[i], 0.f); //offsets[i];
				v_normal = normals[i];
				v_color = colors[i];
				v_lightPosition = v_lightPositionGeometry[0];
				v_texCoords = texCoords[i];
				EmitVertex();
			}

			EndPrimitive();
		}
	)";

	std::string LineBillboardMaterial::s_lineBillboardGeometryShaderWorldSpace = R"(
		#version 430 core
		
		layout(lines) in;
		layout(triangle_strip, max_vertices=4) out;

		uniform mat4 u_worldToView;
		uniform mat4 u_viewToProjection;
		uniform float u_lineWidth;
		
		in vec3[] v_positionGeometry;
		in vec3[] v_normalGeometry;
		in vec4[] v_colorGeometry;
		in vec3[] v_lightPositionGeometry;

		out vec3 v_position;
		out vec3 v_normal;
		out vec4 v_color;
		out vec3 v_lightPosition;
		out vec2 v_texCoords;

		void main() {
			// create a quad that is equivalent to the line, but a quad
			vec4 begin = gl_in[0].gl_Position;
			vec4 end = gl_in[1].gl_Position;
			
			float offsetFactors[4] = {
				-u_lineWidth * 0.5f,
				u_lineWidth * 0.5f,
				-u_lineWidth * 0.15f,
				u_lineWidth * 0.15f,
			};

			vec2 texCoords[4] = {
				vec2(0,0),
				vec2(0,1),
				vec2(1,0),
				vec2(1,1),
			};

			vec4 origins[4] = {
				begin, begin, end, end,
			};

			vec3 positions[4] = {
				v_positionGeometry[0],
				v_positionGeometry[0],
				v_positionGeometry[1],
				v_positionGeometry[1],
			};

			vec3 normals[4] = {
				v_normalGeometry[0],
				v_normalGeometry[0],
				v_normalGeometry[1],
				v_normalGeometry[1],
			};

			vec4 colors[4] = {
				v_colorGeometry[0],
				v_colorGeometry[0],
				v_colorGeometry[1],
				v_colorGeometry[1],
			};


			for (int i = 0; i < 4; ++i) {
				vec3 toCamera = normalize(-positions[i]);
				vec3 lineDirection = normalize(v_positionGeometry[1] - v_positionGeometry[0]);
				vec3 right = normalize(cross(toCamera, lineDirection));
				vec3 positionOffset = right * offsetFactors[i];
				v_position = positions[i] + positionOffset;
				vec4 worldSpacePosition = u_viewToProjection * vec4(v_position, 1.f);
				gl_Position = worldSpacePosition / worldSpacePosition.w;
				v_normal = normals[i];
				v_color = colors[i];
				v_lightPosition = v_lightPositionGeometry[0];
				v_texCoords = texCoords[i];
				
				EmitVertex();
			}

			EndPrimitive();
		}
	)";

	std::string ColoredLineBillboardMaterial::s_lineBillboardFragmentSrc = R"(
		#version 430 core
		
		// o_: output_
		layout(location = 0) out vec4 o_color;

		// u_: uniform_
		uniform vec4 u_color;
		uniform float u_colorTintFactor;

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
		
		in vec2 v_texCoords;
		
		void main() {
			vec4 color = mix(v_color, u_color, u_colorTintFactor);

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

	// ##################################################################### //
	// ### ColoredLineBillboardMaterial #################################### //
	// ##################################################################### //

	Ref<Shader> ColoredLineBillboardMaterial::s_screenSpaceShader = nullptr;
	Ref<Shader> ColoredLineBillboardMaterial::s_worldSpaceShader = nullptr;

	ColoredLineBillboardMaterial::ColoredLineBillboardMaterial()
		: LineBillboardMaterial()
	{
		initialize();
	}

	void ColoredLineBillboardMaterial::initialize() {
		if (s_screenSpaceShader == nullptr) {
			// initialize screen space shader
			std::unordered_map<ShaderRoutine, std::string> sources;
			sources[ShaderRoutine::VertexShader] = BillboardMaterial::s_billboardVertexScreenSpaceSrc;
			sources[ShaderRoutine::GeometryShader] = BillboardMaterial::s_billboardGeometryScreenSpaceSrc;
			sources[ShaderRoutine::FragmentShader] = LineBillboardMaterial::s_lineBillboardFragmentSrc;
			s_screenSpaceShader = Shader::create("ColoredLineBillboardSS", sources);
		}
		if (s_worldSpaceShader == nullptr) {
			// initialize world space shader
			std::unordered_map<ShaderRoutine, std::string> sources;
			sources[ShaderRoutine::VertexShader] = BillboardMaterial::s_billboardVertexWorldSpaceSrc;
			sources[ShaderRoutine::GeometryShader] = LineBillboardMaterial::s_lineBillboardGeometryShaderWorldSpace;
			sources[ShaderRoutine::FragmentShader] = LineBillboardMaterial::s_lineBillboardFragmentSrc;
			s_worldSpaceShader = Shader::create("ColoredLineBillboardWS", sources);
		}
	}

	Ref<Shader> ColoredLineBillboardMaterial::getScreenSpaceBillboardShader() const {
		return s_screenSpaceShader;
	}

	Ref<Shader> ColoredLineBillboardMaterial::getWorldSpaceBillboardShader() const {
		return s_worldSpaceShader;
	}

}
