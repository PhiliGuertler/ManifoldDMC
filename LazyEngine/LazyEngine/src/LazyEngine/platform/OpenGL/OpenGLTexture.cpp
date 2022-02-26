// ######################################################################### //
// ### OpenGLTexture.cpp ################################################### //
// ### implements OpenGLTexture.h                                        ### //
// ######################################################################### //

#include "LazyEngine/gepch.h"
#include "OpenGLTexture.h"

#include "stb_image.h"

#include <glad/glad.h>

#include "LazyEngine/Profiling/Profiler.h"

namespace LazyEngine {

	static inline GLint textureFilterToOpenGL(TextureFilter filter) {
		switch (filter) {
		case TextureFilter::Linear:
			return GL_LINEAR;
		case TextureFilter::Nearest:
			return GL_NEAREST;
		default:
			LAZYENGINE_CORE_ERROR("TextureFilter is unknown in OpenGLTexture!");
			return -1;
		}
	}

	static inline GLint textureWrapToOpenGL(TextureWrap wrap) {
		switch (wrap) {
		case TextureWrap::Repeat:
			return GL_REPEAT;
		case TextureWrap::ClampToEdge:
			return GL_CLAMP_TO_EDGE;
		case TextureWrap::RepeatMirrored:
			return GL_MIRRORED_REPEAT;
		case TextureWrap::MirrorOnce:
			return GL_MIRROR_CLAMP_TO_EDGE;
		default:
			LAZYENGINE_CORE_ERROR("TextureWrap is unknown in OpenGLTexture!");
			return -1;
		}
	}

	static inline GLenum textureFormatToInternalFormat(TextureFormat format) {
		switch (format) {
		case TextureFormat::R:
			return GL_R8;
		case TextureFormat::RG:
			return GL_RG8;
		case TextureFormat::RGB:
			return GL_RGB8;
		case TextureFormat::RGBA:
			return GL_RGBA8;
		case TextureFormat::DepthOnly:
			return GL_DEPTH_COMPONENT24;
		case TextureFormat::DepthStencil:
			return GL_DEPTH24_STENCIL8;
		default:
			LAZYENGINE_CORE_ERROR("TextureFormat is unknown in OpenGLTexture!");
			return -1;
		}
	}

	static inline GLenum textureFormatToDataFormat(TextureFormat format) {
		switch (format) {
		case TextureFormat::R:
			return GL_RED;
		case TextureFormat::RG:
			return GL_RG;
		case TextureFormat::RGB:
			return GL_RGB;
		case TextureFormat::RGBA:
			return GL_RGBA;
		case TextureFormat::DepthOnly:
			return GL_DEPTH_COMPONENT;
		case TextureFormat::DepthStencil:
			return GL_DEPTH_STENCIL;
		default:
			LAZYENGINE_CORE_ERROR("TextureFormat is unknown in OpenGLTexture!");
			return -1;
		}
	}

	OpenGLTexture2D::OpenGLTexture2D(const std::string& filepath)
		: m_width(0)
		, m_height(0)
		, m_rendererID(0)
		, m_internalFormat(GL_NONE)
		, m_dataFormat(GL_NONE)
		, m_path(filepath)
	{
		LAZYENGINE_PROFILE_FUNCTION();

		int width, height, channels;
		// load image file
		stbi_set_flip_vertically_on_load(1);
		stbi_uc* data = stbi_load(filepath.c_str(), &width, &height, &channels, 0);
		// data will be nullptr if the load failed
		if (!data) {
			//LAZYENGINE_CORE_ERROR("Failed to load image: {0}", path.c_str());
			//LAZYENGINE_CORE_ASSERT(false, "Critical Error");
			LAZYENGINE_CORE_ASSERT(false, "Critical Error: Failed to load image: {0}", filepath.c_str());
		}
		// update dimensions of the texture
		m_height = height;
		m_width = width;
		LAZYENGINE_CORE_INFO("Texture {2} loaded with width={0} and height={1}", m_width, m_height, filepath.c_str());

		//GLenum internalFormat = 0;
		//GLenum dataFormat = 0;
		if (channels == 3) {
			// this is a rgb image
			m_internalFormat = GL_RGB8;
			m_dataFormat = GL_RGB;
		}
		else if (channels == 4) {
			// this is a rgba image
			m_internalFormat = GL_RGBA8;
			m_dataFormat = GL_RGBA;
		}
		// TODO: extend me for 16 bit channels and other kinds of textures

		LAZYENGINE_CORE_ASSERT(m_internalFormat && m_dataFormat, "Texture Format not supported.");

		// create 1 new texture
		glCreateTextures(GL_TEXTURE_2D, 1, &m_rendererID);
		// create only 1 mipmap level (for now), using RGB channels with 8 bits each: 
		glTextureStorage2D(m_rendererID, 1, m_internalFormat, m_width, m_height);

		// set up some texture parameters: minification and interpolation on large geometry
		glTextureParameteri(m_rendererID, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTextureParameteri(m_rendererID, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		// repeat the texture in both axes by default for magnification purposes
		glTextureParameteri(m_rendererID, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTextureParameteri(m_rendererID, GL_TEXTURE_WRAP_T, GL_REPEAT);

		// upload texture to level 0 with no x- and y-offset
		glTextureSubImage2D(m_rendererID, 0, 0, 0, m_width, m_height, m_dataFormat, GL_UNSIGNED_BYTE, data);

		// free data from cpu memory, as it is now in gpu memory
		stbi_image_free(data);
	}

	OpenGLTexture2D::OpenGLTexture2D(uint32_t width, uint32_t height, void* data, uint32_t size, TextureFormat format)
		: m_width(width)
		, m_height(height)
		, m_rendererID(0)
		// TODO: extend me for 16 bit channels and other kinds of textures
		, m_internalFormat(textureFormatToInternalFormat(format))
		, m_dataFormat(textureFormatToDataFormat(format))
		, m_path("")
	{
		LAZYENGINE_PROFILE_FUNCTION();

		// create 1 new texture
		glCreateTextures(GL_TEXTURE_2D, 1, &m_rendererID);
		// create only 1 mipmap level (for now), using RGB channels with 8 bits each: 
		glTextureStorage2D(m_rendererID, 1, m_internalFormat, m_width, m_height);

		// set up some texture parameters: minification and interpolation on large geometry
		// TODO: GL_LINEAR, GL_NEAREST, GL_REPEAT, etc should be parameterized!
		glTextureParameteri(m_rendererID, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTextureParameteri(m_rendererID, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		// repeat the texture in both axes by default for magnification purposes
		glTextureParameteri(m_rendererID, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTextureParameteri(m_rendererID, GL_TEXTURE_WRAP_T, GL_REPEAT);

		if (data != nullptr && size != 0) {
			// only upload data if there is any data to begin with
			setData(data, size);
		}
	}

	void OpenGLTexture2D::setData(void* data, uint32_t size) {
		LAZYENGINE_PROFILE_FUNCTION();

		// HACK: this assumes that only GL_RGB and GL_RGBA are in use
		// with an internal format of GL_RGBA8 which is 8 bit per channel
		uint32_t bytesPerPixel = m_dataFormat == GL_RGBA ? 4 : 3;
		// :KCAH

		LAZYENGINE_CORE_ASSERT(size == m_width * m_height * bytesPerPixel, "The data size does not match the texture size! Texture Size: {0}, Input size: {3}.", m_width * m_height * bytesPerPixel, size);
		glTextureSubImage2D(m_rendererID, 0, 0, 0, m_width, m_height, m_dataFormat, GL_UNSIGNED_BYTE, data);
	}

	OpenGLTexture2D::~OpenGLTexture2D() {
		LAZYENGINE_PROFILE_FUNCTION();

		// free gpu memory of this texture
		glDeleteTextures(1, &m_rendererID);
	}

	void OpenGLTexture2D::bind(uint32_t slot) const {
		LAZYENGINE_PROFILE_FUNCTION();

		// bind the wrapped texture to slot which is 0 as a default
		glBindTextureUnit(slot, m_rendererID);
	}

	bool OpenGLTexture2D::operator==(const Texture& other) const {
		LAZYENGINE_PROFILE_FUNCTION();

		try {
			// check if the other texture is also an OpenGLTexture2D
			auto& otherGL = dynamic_cast<const OpenGLTexture2D&>(other);

			// compare the rendererIDs
			return otherGL.m_rendererID == m_rendererID;
		}
		catch (const std::bad_cast& c) {
			// the other texture is of a different type
			LAZYENGINE_CORE_WARN("Comparing two Texture2Ds of different type: \"{0}\"", c.what());
			return false;
		}
	}

	void OpenGLTexture2D::setTextureFilterMinification(TextureFilter filter) {
		glTextureParameteri(m_rendererID, GL_TEXTURE_MIN_FILTER, textureFilterToOpenGL(filter));
	}

	void OpenGLTexture2D::setTextureFilterMagnification(TextureFilter filter) {
		glTextureParameteri(m_rendererID, GL_TEXTURE_MAG_FILTER, textureFilterToOpenGL(filter));
	}

	void OpenGLTexture2D::setTextureWrap(TextureWrap wrap) {
		auto glWrap = textureWrapToOpenGL(wrap);
		glTextureParameteri(m_rendererID, GL_TEXTURE_WRAP_S, glWrap);
		glTextureParameteri(m_rendererID, GL_TEXTURE_WRAP_T, glWrap);
	}

}