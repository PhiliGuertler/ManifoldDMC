#pragma once

// ######################################################################### //
// ### OpenGLTexture.h ##################################################### //
// ### implements Texture.h for OpenGL                                   ### //
// ######################################################################### //

#include "LazyEngine/Renderer/Texture.h"

namespace LazyEngine {

	/**
	 *	An OpenGL implementation of a two dimensional texture
	 */
	class OpenGLTexture2D : public Texture2D {
	public:
		/**
		 *	constructor
		 *	@param filepath: path to the texture file (e.g. a *.png)
		 */
		OpenGLTexture2D(const std::string& filepath);

		/**
		 *	constructor
		 *	@param width: width of the texture in pixels
		 *	@param height: height of the texture in pixels
		 *	@param data: a pointer to the data to be uploaded. If this is nullptr, no data will be uploaded.
		 *	@param size: the size of the data to be uploaded in bytes. If this is 0, no data will be uploaded.
		 */
		OpenGLTexture2D(uint32_t width, uint32_t height, void *data, uint32_t size, TextureFormat format = TextureFormat::RGBA);

		/**
		 *	destructor
		 */
		virtual ~OpenGLTexture2D();

		/**
		 *	returns the width of the texture in pixels.
		 */
		virtual uint32_t getWidth() const override { return m_width; }
		/**
		 *	returns the height of the texture in pixels.
		 */
		virtual uint32_t getHeight() const override { return m_height; }

		/**
		 *	Sets the data of the texture.
		 *	@param data: A pointer to the data.
		 *	@param size: The size of the data to be uploaded in bytes.
		 */
		virtual void setData(void *data, uint32_t size) override;

		/**
		 *	TODO: save the texture slot somehow.
		 *	this should maybe be handled by the shader.
		 *	@param slot: the texture slot to which this texture should be bound.
		 */
		virtual void bind(uint32_t slot = 0) const override;

		/**
		 *	checks if this texture and the other texture are the same
		 */
		virtual bool operator==(const Texture& other) const override;

		virtual void setTextureFilterMinification(TextureFilter filter) override;
		virtual void setTextureFilterMagnification(TextureFilter filter) override;

		virtual void setTextureWrap(TextureWrap wrap) override;

		virtual inline RendererID getRendererID() const override { return m_rendererID; }

	private:
		// the width of the texture in pixels
		uint32_t m_width;
		// the height of the texture in pixels
		uint32_t m_height;
		// the OpenGL handle of the texture
		RendererID m_rendererID;

		// The sized internal format to be used to store the texture data. Examples:
		//	GL_R8, GL_R16, GL_RG8, GL_RG16, GL_RGBA8 and a lot more
		uint32_t m_internalFormat;
		// The format of the pixel data. Might be one of the following:
		//	GL_RED, GL_RG, GL_RGB, GL_BGR, GL_RGBA, GL_BGRA, GL_DEPTH_COMPONENT and GL_STENCIL_INDEX
		uint32_t m_dataFormat;

		// path will be saved for hot reload in debugging.
		// FIXME: maybe this should be outsourced to an asset manager
		std::string m_path;
	};

}