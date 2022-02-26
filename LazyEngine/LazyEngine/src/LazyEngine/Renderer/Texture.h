#pragma once

// ######################################################################### //
// ### Texture.h ########################################################### //
// ### defines texture superclasses that have to be implemented by each  ### //
// ### platform, like OpenGL, Vulcan, ...                                ### //
// ######################################################################### //

#include <string>
#include <array>

#include "LazyEngine/Core/Core.h"
#include <glm/glm.hpp>

namespace LazyEngine {

	/**
	 *	specifies
	 */
	enum class TextureFilter {
		Linear,		// this means that linear interpolation will be used between pixels of a texture
		Nearest,	// this means that the value of the closer pixel will be used
		// TODO: extend for mipmaps
	};

	enum class TextureWrap {
		Repeat,			// endlessly repeats the texture if the texture coordinates leave [0,1]
		ClampToEdge,	// endlessly repeats the last pixel if the texture coordinates leave [0,1]
		RepeatMirrored,	// endlessly repeats the texture while mirroring every second time if the texture coordinates leave [0,1]
		MirrorOnce,		// mirrors the texture once, then ClampToEdge kicks in
	};

	enum class TextureFormat {
		RedOnly,			R = RedOnly,
		RedGreen,			RG = RedGreen,
		RedGreenBlue,		RGB = RedGreenBlue,
		RedGreenBlueAlpha,	RGBA = RedGreenBlueAlpha,
		DepthOnly,
		DepthStencil,
	};

	/**
	 *	The super class of all textures which acts as an interface
	 */
	class Texture {
	public:
		/**
		 *	default constructor
		 */
		virtual ~Texture() = default;

		/**
		 *	returns the width of the texture in pixels.
		 */
		virtual uint32_t getWidth() const = 0;
		/**
		 *	returns the height of the texture in pixels.
		 */
		virtual uint32_t getHeight() const = 0;
		/**
		 *	returns the size of the texture in pixels in [width, height].
		 */
		virtual const glm::ivec2 getSize() const {
			return { getWidth(), getHeight() };
		}

		/**
		 *	Sets the data of the texture.
		 *	@param data: A pointer to the data.
		 *	@param size: The size of the data to be uploaded in bytes.
		 */
		virtual void setData(void* data, uint32_t size) = 0;

		/**
		 *	binds the texture to a given slot.
		 *	@param slot: the target slot that the texture should be bound to
		 */
		virtual void bind(uint32_t slot = 0) const = 0;

		/**
		 *	checks if this texture and the other texture share the same rendererID
		 */
		virtual bool operator==(const Texture& other) const = 0;

		/**
		 *	Sets the interpolation method when drawing the texture smaller than its resolution allows
		 */
		virtual void setTextureFilterMinification(TextureFilter filter) = 0;
		/**
		 *	Sets the interpolation method when drawing the texture bigger than its resolution allows
		 */
		virtual void setTextureFilterMagnification(TextureFilter filter) = 0;

		/**
		 *	Sets the texture wrap mode, which defines the behaviour for texture-coordinates outside of [0,1].
		 */
		virtual void setTextureWrap(TextureWrap wrap) = 0;

		virtual RendererID getRendererID() const = 0;
	};

	/**
	 *	An implementation of a two dimensional texture
	 */
	class Texture2D : public Texture {
	public:
		/**
		 *	creates a two dimensional texture from an asset on disk
		 *	@param filepath: path to the texture file (e.g. a *.png)
		 */
		static Ref<Texture2D> create(const std::string& filepath);

		/**
		 *	creates a two dimensional texture
		 *	@param width: width of the texture in pixels
		 *	@param height: height of the texture in pixels
		 *	@param data: a pointer to the data to be uploaded. If this is nullptr, no data will be uploaded.
		 *	@param size: the size of the data to be uploaded in bytes. If this is 0, no data will be uploaded.
		 */
		static Ref<Texture2D> create(uint32_t width, uint32_t height, void* data = nullptr, uint32_t size = 0, TextureFormat format = TextureFormat::RGBA);

		/**
		 *	creates a two dimensional depth-stencil texture
		 *	@param width: width of the texture in pixels
		 *	@param height: height of the texture in pixels
		 */
		static Ref<Texture2D> createDepthStencil(uint32_t width, uint32_t height);
	};

	/**
	 *	A subtexture is a subsection of an already existing texture
	 */
	class SubTexture2D {
	public:
		/**
		 *	constructor. For easy creation of a Subtexture from a TextureAtlas, see TextureAtlas.h
		 *	@param texture: The texture of which this is a subsection
		 *	@param textureCoordinates: The texture coordinates of the four vertices of a quad.
		 *		The order of vertices is bottom-left, bottom-right, top-right, top-left
		 */
		SubTexture2D(Ref<Texture2D> texture, const std::array<glm::vec2, 4>& textureCoordinates);
		/**
		 *	default destructor
		 */
		~SubTexture2D() = default;

		/**
		 *	returns the texture associated to this sub texture
		 */
		inline Ref<Texture2D> getTexture() const { return m_texture; }
		/**
		 *	returns the texture coordinates of this sub texture.
		 *		The order of vertices is bottom-left, bottom-right, top-right, top-left
		 */
		inline const std::array<glm::vec2, 4>& getTextureCoordinates() const { return m_textureCoordinates; }

	private:
		// the texture of which this sub texture is a subset of
		Ref<Texture2D> m_texture;
		// the texture coordinates of a quad representing the sub-texture slice.
		// The order of vertices is bottom - left, bottom - right, top - right, top - left
		std::array<glm::vec2, 4> m_textureCoordinates;
	};
}