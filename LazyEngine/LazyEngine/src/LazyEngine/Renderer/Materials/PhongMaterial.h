#pragma once

#include "Material.h"

#include "LazyEngine/Renderer/Texture.h"

namespace LazyEngine {

	// ##################################################################### //
	// ### PhongMaterial ################################################### //
	// ##################################################################### //

	/**
	 *	A simple Phong Material that uses Two per-vertex-inputs:
	 *  vec3 a_position
	 *  vec3 a_normal
	 */
	class PhongMaterial : public Material {
	public:
		PhongMaterial();
		virtual ~PhongMaterial() = default;

		/**
		 *	returns the shader that is representing this material
		 */
		virtual inline Ref<Shader> getShader() const override { return PhongMaterial::s_shader; }

		/**
		 *	updates this material's shader's uniforms
		 */
		virtual void updateUniforms() override;

		inline void setAmbientFactor(float ambient) { m_ambientFactor = ambient; }
		inline void setDiffuseFactor(float diffuse) { m_diffuseFactor = diffuse; }
		inline void setSpecularFactor(float specular) { m_specularFactor = specular; }
		inline void setSpecularExponent(float exponent) { m_specularExponent = exponent; }
		inline void setLightPosition(const glm::vec3& position) { m_lightPosition = position; }
		inline void setLightColor(const glm::vec4& lightColor) { m_lightColor = lightColor; }
		inline void setColor(const glm::vec4& color) { m_color = color; }
		inline void setZOffset(float offset) { m_zOffset = offset; }

	protected:
		void updateUniformsForShader(Ref<Shader> shader);

	private:
		void initialize();

	private:
		static Ref<Shader> s_shader;
	
	protected:
		static std::string s_phongVertexShaderSrc;
		static std::string s_phongFragmentShaderSrc;

	protected:
		float m_ambientFactor;
		float m_diffuseFactor;
		float m_specularFactor;
		float m_specularExponent;

		glm::vec3 m_lightPosition;
		glm::vec4 m_lightColor;

		glm::vec4 m_color;

		float m_zOffset;
	};


	// ##################################################################### //
	// ### ColoredPhongMaterial ############################################ //
	// ##################################################################### //

	/**
	 *	A simple Phong Material that uses Three per-vertex-inputs:
	 *  vec3 a_position
	 *  vec3 a_normal
	 *  vec4 a_color
	 */
	class ColoredPhongMaterial : public PhongMaterial {
	public:
		/**
		 *	constructor
		 */
		ColoredPhongMaterial();
		/**
		 *	default destructor
		 */
		virtual ~ColoredPhongMaterial() = default;

		/**
		 *	returns the shader that is representing this material
		 */
		virtual inline Ref<Shader> getShader() const override { return ColoredPhongMaterial::s_shader; }

		inline Ref<Shader> getQuadOutlineShader() const {
			return ColoredPhongMaterial::s_quadOutlineShader;
		}

		inline Ref<Shader> getTriangleOutlineShader() const {
			return ColoredPhongMaterial::s_triangleOutlineShader;
		}

		inline Ref<Shader> getPointShader() const {
			return ColoredPhongMaterial::s_pointShader;
		}

		/**
		 *	updates this material's shader's uniforms
		 */
		virtual void updateUniforms() override;
	private:
		void initialize();

		void initializeQuadOutlineShader();
		void initializeTriangleOutlineShader();
		void initializePointShader();

	private:
		static Ref<Shader> s_shader;
		static Ref<Shader> s_quadOutlineShader;
		static Ref<Shader> s_triangleOutlineShader;
		static Ref<Shader> s_pointShader;

	protected:
		static std::string s_coloredPhongVertexShaderSrc;
		static std::string s_coloredPhongFragmentShaderSrc;

		static std::string s_quadOutlineGeometryShaderSrc;
		static std::string s_triangleOutlineGeometryShaderSrc;
		static std::string s_pointsGeometryShaderSrc;
	};

	// ##################################################################### //
	// ### TexturedPhongMaterial ########################################### //
	// ##################################################################### //

	/**
	 *	A simple Phong Material that uses Three per-vertex-inputs:
	 *  vec3 a_position
	 *  vec3 a_normal
	 *  vec2 a_textureCoordinates
	 */
	class TexturedPhongMaterial : public PhongMaterial {
	public:
		/**
		 *	constructor
		 *	@param texture: the texture that should be used by the material
		 *  @param colorTintFactor: The amount
		 */
		TexturedPhongMaterial(Ref<Texture2D> texture, float colorTintFactor = 0.5f);
		/**
		 *	default destructor
		 */
		virtual ~TexturedPhongMaterial() = default;

		/**
		 *	returns the shader that is representing this material
		 */
		virtual inline Ref<Shader> getShader() const override { return TexturedPhongMaterial::s_shader; }

		/**
		 *	updates this material's shader's uniforms
		 */
		virtual void updateUniforms() override;

		inline void setTexture(Ref<Texture2D> texture) { m_texture = texture; }
		inline void setTintFactor(float tintFactor) { m_colorTintFactor = tintFactor; }
	private:
		void initialize();

	private:
		static Ref<Shader> s_shader;

	protected:
		static std::string s_texturedPhongVertexShaderSrc;
		static std::string s_texturedPhongFragmentShaderSrc;

	protected:
		Ref<Texture2D> m_texture;
		float m_colorTintFactor;

	};




	// ##################################################################### //
	// ### BlinnPhongMaterial ############################################## //
	// ##################################################################### //

	class BlinnPhongMaterial : public PhongMaterial {
	public:
		BlinnPhongMaterial();
		virtual ~BlinnPhongMaterial() = default;

		virtual inline Ref<Shader> getShader() const override { return BlinnPhongMaterial::s_shader; }

		virtual void updateUniforms() override;

	private:
		void initialize();

	protected:
		static Ref<Shader> s_shader;

		static std::string s_blinnPhongFragmentShaderSrc;
	};



	// ##################################################################### //
	// ### TexturedBlinnPhongMaterial ###################################### //
	// ##################################################################### //

	class TexturedBlinnPhongMaterial : public TexturedPhongMaterial {
	public:
		TexturedBlinnPhongMaterial(Ref<Texture2D> texture, float colorTintFactor = 0.5f);
		virtual ~TexturedBlinnPhongMaterial() = default;

		virtual inline Ref<Shader> getShader() const override { return TexturedBlinnPhongMaterial::s_shader; }

		virtual void updateUniforms() override;

	private:
		void initialize();

	protected:
		static Ref<Shader> s_shader;

		static std::string s_texturedBlinnPhongFragmentShaderSrc;
	};

}
