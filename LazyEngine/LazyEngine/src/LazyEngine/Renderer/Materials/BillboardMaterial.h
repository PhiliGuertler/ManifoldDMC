#pragma once

#include "LazyEngine/Core/Core.h"
#include "LazyEngine/Renderer/Texture.h"
#include "PhongMaterial.h"

namespace LazyEngine {

	/**
	 *	Creates a Camera-aligned Billboard using geometry shaders
	 *	TODO: FIXME: The Screen-Space variants are not working correctly yet!
	 */
	class BillboardMaterial : public PhongMaterial {
	public:
		BillboardMaterial();
		virtual ~BillboardMaterial();

		/**
		 *	returns the shader that is representing this material
		 */
		virtual Ref<Shader> getShader() const;

		/**
		 *	updates this material's shader's uniforms
		 */
		virtual void updateUniforms() override;

		/**
		 *	Sets the width of the billboard in world space coordinates
		 */
		void setWorldSpaceWidth(float width);
		/**
		 *	Sets the height of the billboard in world space coordinates
		 */
		void setWorldSpaceHeight(float height);
		/**
		 *	Sets both width and height of the billboard in world space coordinates
		 */
		void setWorldSpaceSize(const glm::vec2& size);

		/**
		 *	Sets the width of the billboard in screen space coordinates
		 */
		void setScreenSpaceWidth(float width);
		/**
		 *	Sets the height of the billboard in screen space coordinates
		 */
		void setScreenSpaceHeight(float height);
		/**
		 *	Sets both width and height of the billboard in screen space coordinates
		 */
		void setScreenSpaceSize(const glm::vec2& size);

		/**
		 *	Sets whether to use screen-space or world-space sizes for the billboard.
		 *	@param value: If true, screen space size will be used, otherwise world space size.
		 */
		void setUseScreenSpaceSize(bool value);

		/**
		 *	Sets a mix-color that will be used in addition to a vertex's color
		 */
		void setColor(const glm::vec4& color);

		/**
		 *	Sets the tint factor, that will determine, how strong the influence of the mix-color
		 *	is supposed to be (0.f by default)
		 */
		void setColorTintFactor(float factor);

	protected:
		virtual Ref<Shader> getScreenSpaceBillboardShader() const = 0;
		virtual Ref<Shader> getWorldSpaceBillboardShader() const = 0;

	protected:
		static std::string s_billboardVertexWorldSpaceSrc;
		static std::string s_billboardVertexScreenSpaceSrc;
		static std::string s_billboardGeometryScreenSpaceSrc;
		static std::string s_billboardGeometryWorldSpaceSrc;

	protected:
		glm::vec2 m_worldSpaceSize;
		glm::vec2 m_screenSpaceSize;
		// If true, screen space size will be used, otherwise world space size will be used
		bool m_useScreenSpaceSize;
		glm::vec4 m_color;
		float m_colorTintFactor;
	};

	/**
	 *	A Billboard Shader that creates Billboards from Vertices.
	 *	The Vertex's color will be used to create a Sphere-like billboard
	 *	The Vertex Layout should be:
	 *	{
	 *		a_position: vec3,
	 *		a_normal: vec3,
	 *		a_color: vec4,
	 *	}
	 */
	class ColoredSphereBillboardMaterial : public BillboardMaterial {
	public:
		ColoredSphereBillboardMaterial();
		virtual ~ColoredSphereBillboardMaterial() = default;

	public:
		static std::string s_coloredSphereBillboardFragmentSrc;

	protected:
		virtual Ref<Shader> getScreenSpaceBillboardShader() const override;
		virtual Ref<Shader> getWorldSpaceBillboardShader() const override;

		void initialize();

	private:
		static Ref<Shader> s_screenSpaceShader;
		static Ref<Shader> s_worldSpaceShader;
	};

	/**
	 *	UNTESTED!
	 */
	class TexturedBillboardMaterial : public BillboardMaterial {
	public:
		TexturedBillboardMaterial(const Ref<Texture2D>& texture);
		virtual ~TexturedBillboardMaterial() = default;

		void setTexture(const Ref<Texture2D>& texture);

		void setTintFactor(float tintFactor);

		virtual void updateUniforms() override;

	public:
		static std::string s_texturedBillboardFragmentSrc;

	protected:
		virtual Ref<Shader> getScreenSpaceBillboardShader() const override;
		virtual Ref<Shader> getWorldSpaceBillboardShader() const override;

		void initialize();

	private:
		static Ref<Shader> s_screenSpaceShader;
		static Ref<Shader> s_worldSpaceShader;
	
	protected:
		Ref<Texture2D> m_texture;
		float m_tintFactor;
	};


	class LineBillboardMaterial : public BillboardMaterial {
	public:
		LineBillboardMaterial();
		virtual ~LineBillboardMaterial() = default;

		virtual void updateUniforms() override;

		inline void setLineWidth(float width) {
			m_lineWidth = width;
		}

	public:
		static std::string s_lineBillboardGeometryShaderWorldSpace;
		static std::string s_lineBillboardGeometryShaderScreenSpace;
		static std::string s_lineBillboardFragmentSrc;

	protected:
		float m_lineWidth;
	};

	/**
	 *	A Billboard shader that creates quads for lines.
	 *	These lines will use the correct perspective.
	 */
	class ColoredLineBillboardMaterial : public LineBillboardMaterial {
	public:
		ColoredLineBillboardMaterial();
		virtual ~ColoredLineBillboardMaterial() = default;

	protected:
		virtual Ref<Shader> getScreenSpaceBillboardShader() const override;
		virtual Ref<Shader> getWorldSpaceBillboardShader() const override;

		void initialize();

	private:
		static Ref<Shader> s_screenSpaceShader;
		static Ref<Shader> s_worldSpaceShader;
	};

}