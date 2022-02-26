#pragma once

#include "LazyEngine/Renderer/Framebuffer.h"

#include "LazyEngine/Renderer/Texture.h"

namespace LazyEngine {

	class OpenGLFramebuffer : public Framebuffer {
	public:
		/**
		 *	constructor
		 */
		OpenGLFramebuffer(uint32_t width, uint32_t height);
		/**
		 *	destructor
		 */
		virtual ~OpenGLFramebuffer();

		/**
		 *	binds the framebuffer
		 */
		virtual void bind(FramebufferUsage usage) override;
		/**
		 *	unbinds the framebuffer
		 */
		virtual void unbind() override;

		/**
		 *	returns the color texture into which this framebuffer is rendering
		 */
		virtual inline Ref<Texture2D> getColorTexture() const override { return m_colorTexture; }

		/**
		 *	returns the depth-stencil texture into which this framebuffer is rendering
		 */
		virtual inline Ref<Texture2D> getDepthStencil() const override { return m_depthStencilTexture; }

		/**
		 *	returns the size of the framebuffer in width and height
		 */
		virtual const glm::ivec2 getSize() const override;

		/**
		 *	resizes the attachments of this framebuffer
		 */
		virtual void resize(uint32_t width, uint32_t height) override;
	private:
		// the opengl handle to this framebuffer
		uint32_t m_rendererID;
		FramebufferUsage m_usage;

		Ref<Texture2D> m_colorTexture;
		Ref<Texture2D> m_depthStencilTexture;

	};

}