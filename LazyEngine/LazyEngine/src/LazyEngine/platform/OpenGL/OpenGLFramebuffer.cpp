#include "LazyEngine/gepch.h"

#include "OpenGLFramebuffer.h"

#include "OpenGLTexture.h"

#include "glad/glad.h"

namespace LazyEngine {

	static inline GLenum framebufferUsageToOpenGL(Framebuffer::FramebufferUsage usage) {
		switch (usage) {
		case Framebuffer::FramebufferUsage::Draw:
			return GL_DRAW_FRAMEBUFFER;
		case Framebuffer::FramebufferUsage::Read:
			return GL_READ_FRAMEBUFFER;
		case Framebuffer::FramebufferUsage::ReadAndDraw:
			return GL_FRAMEBUFFER;
		default:
			return 0;
		}
	}

	OpenGLFramebuffer::OpenGLFramebuffer(uint32_t width, uint32_t height)
		: m_rendererID(0)
		, m_usage(FramebufferUsage::ReadAndDraw)
		, m_colorTexture(nullptr)
		, m_depthStencilTexture(nullptr)
	{
		// create an opengl framebuffer
		glGenFramebuffers(1, &m_rendererID);

		resize(width, height);
	}

	OpenGLFramebuffer::~OpenGLFramebuffer() {
		// destroy the opengl framebuffer
		glDeleteFramebuffers(1, &m_rendererID);
	}

	void OpenGLFramebuffer::bind(FramebufferUsage usage) {
		// bind this framebuffer
		m_usage = usage;
		GLenum target = framebufferUsageToOpenGL(usage);
		glBindFramebuffer(target, m_rendererID);

		// set the gl viewport to the size of this framebuffer
		glViewport(0,0, m_colorTexture->getWidth(), m_colorTexture->getHeight());
	}

	void OpenGLFramebuffer::unbind() {
		// bind the default framebuffer (the screen)
		glBindFramebuffer(framebufferUsageToOpenGL(m_usage), 0);
	}

	const glm::ivec2 OpenGLFramebuffer::getSize() const {
		return m_colorTexture->getSize();
	}

	void OpenGLFramebuffer::resize(uint32_t width, uint32_t height) {
		// create the textures that will be used for this framebuffer
		// create the color texture
		m_colorTexture = Texture2D::create(width, height);
		m_colorTexture->setTextureFilterMagnification(TextureFilter::Linear);
		m_colorTexture->setTextureWrap(TextureWrap::ClampToEdge);
		auto openglTexture = std::static_pointer_cast<OpenGLTexture2D>(m_colorTexture);

		// bind the framebuffer
		bind(m_usage);
		GLenum usage = framebufferUsageToOpenGL(m_usage);

		// attach the color texture
		glFramebufferTexture2D(usage, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, openglTexture->getRendererID(), 0);

		// create the depth-stencil texture
		m_depthStencilTexture = Texture2D::createDepthStencil(width, height);
		auto openglDepthTexture = std::static_pointer_cast<OpenGLTexture2D>(m_depthStencilTexture);
		
		// attach the depth stencil texture
		glFramebufferTexture2D(usage, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, openglDepthTexture->getRendererID(), 0);

		// check if the framebuffer encountered an error
		if (glCheckNamedFramebufferStatus(m_rendererID, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			LAZYENGINE_CORE_ERROR("Framebuffer is incomplete!");
		}

		unbind();
	}

}